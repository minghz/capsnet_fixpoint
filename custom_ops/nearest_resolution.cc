#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

REGISTER_OP("NearestResolution")
.Input("to_nearest: float") //input tensor
.Input("range_bits: int32") // range and precision bits (m, n)
.Input("precision_bits: int32") // range and precision bits (m, n)
.Output("nearested: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
});

class NearestResolutionOp : public OpKernel {
  public:
    explicit NearestResolutionOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& input_tensor = context->input(0);
      const Tensor& range_bits = context->input(1);
      const Tensor& precision_bits = context->input(2);

      auto input = input_tensor.flat<float>();
      auto m = range_bits.flat<int>();
      auto n = precision_bits.flat<int>();
     
      float range = pow(2, (m(0) - 1));
      float resolution = pow(2, -1 * n(0));
      float range_min = -1 * range;
      float range_max = range - resolution;

      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, input_tensor.shape(), &output_tensor));
      auto output = output_tensor->flat<float>();

      // convert input tensor to nearested point equivalent range
      const int input_count = input.size();
      for (int i = 0; i < input_count; i++) {

        // clip on max and min of allowed range
        if (input(i) > range_max || input(i) < range_min ) {
          if (input(i) > range_max) { output(i) = range_max; }
          if (input(i) < range_min) { output(i) = range_min; }

        // convert resolution to nearested point equivalent
        } else {
          float nearest_equivalent = resolution * round(input(i) / resolution);
          output(i) = nearest_equivalent;
        }
      }
    }
};

REGISTER_OP("NearestResolutionGrad")
  .Input("grad: float") //input tensor
  .Input("to_nearest: float") //input tensor
  .Input("range_bits: int32") // range and precision bits (m, n)
  .Input("precision_bits: int32") // range and precision bits (m, n)
  .Output("nearested_grad: float")
  .Output("range_bits_grad: int32") // range and precision bits (m, n)
  .Output("precision_bits_grad: int32") // range and precision bits (m, n)
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
});

class NearestResolutionGradOp : public OpKernel {
  public:
    explicit NearestResolutionGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& gradient = context->input(0);
      auto input = gradient.flat<float>();

      const Tensor& range_bits = context->input(1);
      const Tensor& precision_bits = context->input(2);

      // Gradient output
      Tensor* nearested_grad = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, gradient.shape(), &nearested_grad));
      auto output = nearested_grad->flat<float>();

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

REGISTER_KERNEL_BUILDER(Name("NearestResolution").Device(DEVICE_CPU), NearestResolutionOp);
REGISTER_KERNEL_BUILDER(Name("NearestResolutionGrad").Device(DEVICE_CPU), NearestResolutionGradOp);
