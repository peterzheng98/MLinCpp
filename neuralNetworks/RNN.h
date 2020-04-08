//
// Created by 郑文鑫 on 2019/12/4.
//

#ifndef MLINCPP_RNN_H
#define MLINCPP_RNN_H
#include "matrix.h"
#include "matrixDef.h"
#include "model.h"
#include <random>
#include <functional>
#include <ctime>
namespace peterzheng {
namespace model {
class RNN : public model {
private:
  std::default_random_engine randomEngine;
  std::uniform_real_distribution<float> uniformRealDistribution;
  std::uniform_int_distribution<unsigned> uniformIntDistribution;
  matrix::matrix<float> x, y;
  matrix::matrix<float> hidden;
  struct __param_matrix {
    matrix::matrix<float> input_weight, output_weight, hidden_weight;
    std::default_random_engine randomEngine;
    std::uniform_real_distribution<float> uniformRealDistribution;
    std::uniform_int_distribution<unsigned> uniformIntDistribution;
    __param_matrix(const matrix::matrix<float> &inputWeight,
                   const matrix::matrix<float> &outputWeight,
                   const matrix::matrix<float> &hiddenWeight);
    __param_matrix() : randomEngine(time(0)) {}
    void randomized(const int& dim_i, const int& dim_h, const int& dim_o);
  };
  __param_matrix paramMatrix;
  std::function<float(const matrix::matrix<float> &,
                      const matrix::matrix<float> &)>
      loss_function;


private:
  float learningRate = 0.01;
  int epoches;
  int hiddenSize, inputSize, outputSize;
public:
  RNN(const matrix::matrix<float> &x, const matrix::matrix<float> &y,
      const matrix::matrix<float> &hidden,
      const std::function<float(const matrix::matrix<float> &,
                                const matrix::matrix<float> &)> &lossFunction,
      float learningRate, int epoches, int hiddenSize, int inputSize,
      int outputSize);

private:
  void internalRun(const int& target_x, const int& target_y, const int& dim = 0, const float& lr = 0.01);

public:
  void run() override;
  void compile() override;
  float loss() override;
};
} // namespace model
} // namespace peterzheng

#endif // MLINCPP_RNN_H
