#include <stdio.h>
#include <iostream>
#include <math.h>

float rstoc(float x) {
  float decimal = abs(x - trunc(x));

  float random_selector = (float)rand() / RAND_MAX;

  float adjustor;
  if (random_selector < decimal) adjustor = 1;
  else adjustor = 0;

  // consider sign
  if(x < 0) adjustor = -1 * adjustor;

  return trunc(x) + adjustor;
}

int main (void) {
  float sum = 0.0;
  for(int i = 0; i < 100; i++) {
    sum += rstoc(0.3);
  }
  std::cout << "sum: " << sum << std::endl;
}
