#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

REGISTER_OP("FloorResolution")
.Input("to_floor: float") //input tensor
.Input("range_bits: int32") // range and precision bits (m, n)
.Input("precision_bits: int32") // range and precision bits (m, n)
.Output("floored: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
});

class FloorResolutionOp : public OpKernel {
  public:
    explicit FloorResolutionOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& input_tensor = context->input(0);
      const Tensor& range_bits = context->input(1);
      const Tensor& precision_bits = context->input(2);

      auto input = input_tensor.flat<float>();
      auto m = range_bits.flat<int>();
      auto n = precision_bits.flat<int>();
      //std::cout << "m0: " << m(0) << std::endl;
      //std::cout << "n0: " << n(0) << std::endl;
     
      float range = pow(2, (m(0) - 1));
      float resolution = pow(2, -1 * n(0));
      float range_min = -1 * range;
      float range_max = range - resolution;

      //std::cout << "range: [" << range_min << ", " << range_max << "]"
      //  << " | resolution: " << resolution
      //  << std::endl;

      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, input_tensor.shape(), &output_tensor));
      auto output = output_tensor->flat<float>();

      // convert input tensor to floored point equivalent range
      // and precision with a 5% resolution tolerance
      const int input_count = input.size();
      for (int i = 0; i < input_count; i++) {

        // clip on max and min of allowed range
        if (input(i) > range_max || input(i) < range_min ) {
          if (input(i) > range_max) { output(i) = range_max; }
          if (input(i) < range_min) { output(i) = range_min; }

        // convert resolution to floored point equivalent
        } else {
          float floor_equivalent = resolution * floor(input(i) / resolution);
          //float deviation_from_orig = abs(floor_equivalent - input(i)) / abs(input(i));
          output(i) = floor_equivalent;
          //if(deviation_from_orig > 0.05) { // more than 5% deviation
          //}
        }
      }
      //std::cout << "%over: " << accuracy(0)
      //  << " %under: " << accuracy(1) << std::endl;
    }
};

REGISTER_OP("FloorResolutionGrad")
  .Input("grad: float") //input tensor
  .Input("to_floor: float") //input tensor
  .Input("range_bits: int32") // range and precision bits (m, n)
  .Input("precision_bits: int32") // range and precision bits (m, n)
  .Output("floored_grad: float")
  .Output("range_bits_grad: int32") // range and precision bits (m, n)
  .Output("precision_bits_grad: int32") // range and precision bits (m, n)
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
});

class FloorResolutionGradOp : public OpKernel {
  public:
    explicit FloorResolutionGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& gradient = context->input(0);
      auto input = gradient.flat<float>();

      const Tensor& range_bits = context->input(1);
      const Tensor& precision_bits = context->input(2);

      // Gradient output
      Tensor* floored_grad = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, gradient.shape(), &floored_grad));
      auto output = floored_grad->flat<float>();

      // Return None grad
      Tensor* range_bits_grad = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(1, range_bits.shape(), &range_bits_grad));

      // Return None grad
      Tensor* precision_bits_grad = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(2, precision_bits.shape(), &precision_bits_grad));

      // For our gradient we are simply seting input = output
      const int input_count = input.size();
      for (int i = 0; i < input_count; i++) {

        output(i) = input(i);

      }
    }
};

REGISTER_KERNEL_BUILDER(Name("FloorResolution").Device(DEVICE_CPU), FloorResolutionOp);
REGISTER_KERNEL_BUILDER(Name("FloorResolutionGrad").Device(DEVICE_CPU), FloorResolutionGradOp);