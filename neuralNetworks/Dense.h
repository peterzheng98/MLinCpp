//
// Created by 郑文鑫 on 2019/12/20.
//

#ifndef MLINCPP_DENSE_H
#define MLINCPP_DENSE_H

#include "matrix.h"
#include "model.h"
#include "parameter.h"
#include <functional>
#include <random>
#include <ctime>
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

public:
  const matrix::matrix<float> &getBias() const;
  activationFunction::type getActivation() const;

public:
  const matrix::matrix<float> &getWeight() const;

private:
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
  const std::function<matrix::matrix<float>(matrix::matrix<float>)> &
  getActivationfunc() const;
  void setWeight(const matrix::matrix<float> &weight);

public:
  void run() override;
  matrix::matrix<float> run(const matrix::matrix<float> &x);
  void compile() override;
  const std::function<matrix::matrix<float>(matrix::matrix<float>)> &
  getActivationGrad() const;
  float loss() override;
  void init();
  matrix::matrix<float> eval(const matrix::matrix<float> &x);
  const matrix::matrix<float> &getInput() const;
  void setInput(const matrix::matrix<float> &input);
  const matrix::matrix<float> &getOutput() const;
  void update(const float &eta, const matrix::matrix<float> &grad);
  const matrix::matrix<float> operator()(const matrix::matrix<float> &x) {
    return run(x);
  }
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
  std::vector<int> config;
  float learningRate;

  lossFunction::type lossfunction;
  std::function<matrix::matrix<float>(matrix::matrix<float>,
                                      matrix::matrix<float>)>
      lossFunction;

  std::string savingPrefix;
  int savingInterval;
  int epoches;
  int totalNum;
  int batchSize;
  static std::random_device r;
  std::default_random_engine randomEngine;
  std::uniform_real_distribution<float> uniformRealDistribution;
  std::uniform_int_distribution<unsigned> uniformIntDistribution;

public:
  Dense(const peterzheng::matrix::matrix<float> &x,
        const peterzheng::matrix::matrix<float> &y, const int &feature,
        const int &samples, const std::vector<int> &layerConnection,
        float learningRate = 0.01,
        peterzheng::model::lossFunction::type lossfunction =
            lossFunction::type::MSE,
        const std::string &savingPrefix = "data/training",
        int savingInterval = 1, int epoches = 128, int batchSize = 64);

public:
  void run() override;
  void compile() override;
  void summary();
  float loss() override;

public:
  void outputProgressBarFlush(int currentEpoch, int slice, int sliceAll,
                              float timePerSlice);
  void outputProgress(int currentEpoch, float timePerEpoch);
};

} // namespace model
} // namespace peterzheng

#endif // MLINCPP_DENSE_H
