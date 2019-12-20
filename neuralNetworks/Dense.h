//
// Created by 郑文鑫 on 2019/12/20.
//

#ifndef MLINCPP_DENSE_H
#define MLINCPP_DENSE_H

#include "../matrix/matrix.h"
#include "../matrix/matrixTools.h"
#include "../toplevel/model.h"
#include "parameter.h"
#include <random>

namespace peterzheng {
namespace model {
// Toy Dense Layer
class DenseCell : public model {
private:
  // function: output = f(weight * input + bias)
  matrix::matrix<float> input, output, weight, bias;
  int outputSize;
  activationFunction::type activation;
  std::function<matrix::matrix<float>(matrix::matrix<float>)> activationfunc;
  std::function<matrix::matrix<float>(matrix::matrix<float>)> activationGrad;

  static std::random_device r;
  std::default_random_engine randomEngine;
  std::uniform_real_distribution<float> uniformRealDistribution;

public:
  DenseCell(const matrix::matrix<float> &input,
            const matrix::matrix<float> &output, const int &outputSize,
            activationFunction::type activation,
            const std::function<matrix::matrix<float>(matrix::matrix<float>)>
                &activationfunc,
            const std::function<matrix::matrix<float>(matrix::matrix<float>)>
                &activationGrad);

public:
  void run() override;
  void compile() override;
  float loss() override;
  void init();
  const matrix::matrix<float> &getInput() const;
  void setInput(const matrix::matrix<float> &input);
  const matrix::matrix<float> &getOutput() const;
  void setOutput(const matrix::matrix<float> &output);
  void update(const float &eta, const matrix::matrix<float> &grad);
};

// Implement as toy multi-dense layer
// Input -> DenseCell(?) -> ... -> DenseCell(?) -> DenseCell(1, as output)
// back propagation
// partial(L)/partial(W)
// To be thought: calculation on joint loss
class Dense : public model {
private:
  matrix::matrix<float> x, y;
  std::vector<DenseCell> kernel;
  float learningRate;

  lossFunction::type lossfunction;
  std::function<matrix::matrix<float>(matrix::matrix<float>,
                                      matrix::matrix<float>)>
      lossFunction;

  std::string savingPrefix;
  int savingInterval;
  int epoches;
  int totalNum;
  static std::random_device r;
  std::default_random_engine randomEngine;
  std::uniform_real_distribution<float> uniformRealDistribution;

public:
  Dense(const matrix::matrix<float> &x, const matrix::matrix<float> &y,
        const int &feature, const int &samples,
        const std::vector<int> &layerConnection, float learningRate,
        lossFunction::type lossfunction,
        const std::function<matrix::matrix<float>(
            matrix::matrix<float>, matrix::matrix<float>)> &lossFunction,
        const std::string &savingPrefix, int savingInterval, int epoches);
};

} // namespace model
} // namespace peterzheng

#endif // MLINCPP_DENSE_H
