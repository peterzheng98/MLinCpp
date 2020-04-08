//
// Created by 郑文鑫 on 2019/12/6.
//

#ifndef MLINCPP_CONV2D_H
#define MLINCPP_CONV2D_H
#include "model.h"
#include "matrixDef.h"
#include "matrix.h"

namespace peterzheng{
namespace model{
class Conv2D : public model{
private:
  matrix::matrix<float> filter, x, y;

public:
  Conv2D(const matrix::matrix<float> &filter, const matrix::matrix<float> &x);

private:
  void run() override;
  void compile() override;
  float loss() override;
};
}
}


#endif // MLINCPP_CONV2D_H
