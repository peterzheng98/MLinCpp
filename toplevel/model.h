//
// Created by 郑文鑫 on 2019/12/3.
//

#ifndef MLINCPP_MODEL_H
#define MLINCPP_MODEL_H

namespace peterzheng{
namespace model{
class model{
  virtual void run() = 0;
  virtual void compile() = 0;
  virtual float loss() = 0;
};
}
}

#endif // MLINCPP_MODEL_H
