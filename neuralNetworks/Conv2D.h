//
// Created by 郑文鑫 on 2019/12/6.
//

#ifndef MLINCPP_CONV2D_H
#define MLINCPP_CONV2D_H
#include "../toplevel/model.h"
#include "../matrix/matrixDef.h"
#include "../matrix/matrix.h"

namespace peterzheng{
namespace model{
class Conv2D : public model{
private:
  matrix::matrix<float> filter, x, y;
  struct __param{

  };
};
}
}


#endif // MLINCPP_CONV2D_H
