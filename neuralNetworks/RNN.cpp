//
// Created by 郑文鑫 on 2019/12/4.
//

#include <ctime>
#include "RNN.h"
void peterzheng::model::RNN::run() {}
void peterzheng::model::RNN::compile() {
  // Setup: Generate the initial matrix
  this->paramMatrix.randomized(inputSize, hiddenSize, outputSize);

}
float peterzheng::model::RNN::loss() { return loss_function(x, y); }
peterzheng::model::RNN::RNN(
    const peterzheng::matrix::matrix<float> &x,
    const peterzheng::matrix::matrix<float> &y,
    const peterzheng::matrix::matrix<float> &hidden,
    const std::function<float(const matrix::matrix<float> &,
                              const matrix::matrix<float> &)> &lossFunction,
    float learningRate, int epoches, int hiddenSize, int inputSize,
    int outputSize)
    : x(x), y(y), hidden(hidden), loss_function(lossFunction),
      learningRate(learningRate), epoches(epoches), hiddenSize(hiddenSize),
      inputSize(inputSize), outputSize(outputSize), randomEngine(time(0)) {}

peterzheng::model::RNN::__param_matrix::__param_matrix(
    const peterzheng::matrix::matrix<float> &inputWeight,
    const peterzheng::matrix::matrix<float> &outputWeight,
    const peterzheng::matrix::matrix<float> &hiddenWeight)
    : input_weight(inputWeight), output_weight(outputWeight),
      hidden_weight(hiddenWeight) {}


void peterzheng::model::RNN::__param_matrix::randomized(const int &dim_i,
                                                        const int &dim_h,
                                                        const int &dim_o) {
  this->input_weight.reload(dim_i, dim_h);
  this->hidden_weight.reload(dim_h, dim_h);
  this->output_weight.reload(dim_h, dim_o);

  // Generating the random matrix
  for(size_t row = 0; row < dim_i; ++row)
    for(size_t col = 0; col < dim_h; ++col)
      this->input_weight(row, col) = uniformRealDistribution(randomEngine);
  for(size_t row = 0; row < dim_h; ++row)
    for(size_t col = 0; col < dim_h; ++col)
      this->hidden_weight(row, col) = uniformRealDistribution(randomEngine);
  for(size_t row = 0; row < dim_h; ++row)
    for(size_t col = 0; col < dim_o; ++col)
      this->output_weight(row, col) = uniformRealDistribution(randomEngine);
}