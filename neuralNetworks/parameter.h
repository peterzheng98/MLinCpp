//
// Created by 郑文鑫 on 2019/12/20.
//

#ifndef MLINCPP_PARAMETER_H
#define MLINCPP_PARAMETER_H

#include "../matrix/matrix.h"
namespace peterzheng {
namespace model {
namespace activationFunction {
enum type { Sigmoid, Tanh, ReLU, Linear, UserDefined };
}
namespace lossFunction {
enum type { MSE, UserDefined };
} // namespace lossFunction
} // namespace model
} // namespace peterzheng

#endif // MLINCPP_PARAMETER_H
