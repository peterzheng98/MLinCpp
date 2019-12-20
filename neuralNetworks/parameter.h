//
// Created by 郑文鑫 on 2019/12/20.
//

#ifndef MLINCPP_PARAMETER_H
#define MLINCPP_PARAMETER_H

namespace peterzheng {
namespace model {
namespace activationFunction {
enum type { Sigmoid, Tanh, ReLU, Linear, UserDefined };
}
namespace lossFunction {
enum type { MSE, UserDefined };
matrix::matrix<float> mse(const matrix::matrix<float> &input,
                          const matrix::matrix<float> &__standard) {
  if (input.getM() != 1 && input.getN() != 1)
    throw exception("MSE cannot applied to matrix object",
                    std::string(__FILE__), "Internal Error", __LINE__);
  auto internal = input;
  if (input.getM() == 1)
    internal.transpose();
  if (__standard.getM() != 1 && __standard.getN() != 1)
    throw exception("MSE cannot applied to matrix object",
                    std::string(__FILE__), "Internal Error", __LINE__);
  auto standard = __standard;
  if (__standard.getM() == 1)
    standard.transpose();
  matrix::matrix<float> ret(internal.getM(), internal.getN());
  for (size_t idx = 0; idx < input.getM(); ++idx) {
    ret(idx, 0) = 0.5 * (standard(idx, 0) - internal(idx, 0)) *
                  (standard(idx, 0) - internal(idx, 0));
  }
  return ret;
}
} // namespace lossFunction
} // namespace model
} // namespace peterzheng

#endif // MLINCPP_PARAMETER_H
