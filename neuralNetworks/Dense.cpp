//
// Created by 郑文鑫 on 2019/12/20.
//

#include "Dense.h"
#include "../matrix/matrixTools.h"
peterzheng::model::DenseCell::DenseCell(
    const peterzheng::matrix::matrix<float> &input,
    const peterzheng::matrix::matrix<float> &output, const int &outputSize,
    peterzheng::model::activationFunction::type activation =
        activationFunction::Sigmoid,
    const std::function<matrix::matrix<float>(matrix::matrix<float>)>
        &activationfunc = peterzheng::matrix::sigmoid<float>,
    const std::function<matrix::matrix<float>(matrix::matrix<float>)>
        &activationGrad = peterzheng::matrix::sigmoidGrad<float>)
    : input(input), output(output), outputSize(outputSize),
      activation(activation), activationfunc(activationfunc),
      activationGrad(activationGrad) {}
void peterzheng::model::DenseCell::run() {
  output = activationfunc(weight * input + bias);
}
void peterzheng::model::DenseCell::compile() {
  if (input.getM() != 1 && input.getN() != 1)
    throw exception("Dense layer request for vector input.",
                    std::string(__FILE__), "Internal Error", __LINE__);
  if (input.getM() == 1)
    input.transpose();
  // hidden_output = f(Wx+b)
  // W: outputSize * m
  // x: m * 1             ----> outputSize * 1
  // b: outputSize * 1
}
float peterzheng::model::DenseCell::loss() {
  throw exception("No loss available for dense cell", std::string(__FILE__),
                  "Internal Error", __LINE__);
}

void peterzheng::model::DenseCell::init() {
  weight.reload(outputSize, input.getM());
  bias.reload(outputSize, 1);
  output.reload(outputSize, 1); // Hidden States
  for (size_t idx1 = 0; idx1 < outputSize; ++idx1) {
    for (size_t idx2 = 0; idx2 < input.getM(); ++idx2) {
      weight(idx1, idx2) = uniformRealDistribution(randomEngine);
    }
    bias(idx1, 0) = uniformRealDistribution(randomEngine);
    output(idx1, 0) = 0.0;
  }
}

void peterzheng::model::DenseCell::update(const float &eta,
                                          const matrix::matrix<float> &grad) {
  weight = weight - grad * eta;
}
const peterzheng::matrix::matrix<float> &
peterzheng::model::DenseCell::getInput() const {
  return input;
}
void peterzheng::model::DenseCell::setInput(
    const peterzheng::matrix::matrix<float> &input) {
  DenseCell::input = input;
}
const peterzheng::matrix::matrix<float> &
peterzheng::model::DenseCell::getOutput() const {
  return output;
}
const peterzheng::matrix::matrix<float> &
peterzheng::model::DenseCell::getWeight() const {
  return weight;
}
const peterzheng::matrix::matrix<float> &
peterzheng::model::DenseCell::getBias() const {
  return bias;
}
void peterzheng::model::Dense::run() {}
void peterzheng::model::Dense::summary() {
  for (size_t idx = 0; idx < 50; idx++)
    std::cout << "=";
  std::cout << std::endl << "Summary" << std::endl;
  for (size_t idx = 0; idx < 50; idx++)
    std::cout << "=";
  std::cout << std::endl;
  int LayerIdx = 0;
  long long sum = 0;
  for (auto &j : kernel) {
    std::cout << "Dense_" << LayerIdx << "\t\t\t(" << j.getWeight().getM()
              << "," << j.getWeight().getN() << ")\t\t\t"
              << j.getWeight().getM() * j.getWeight().getN() +
                     j.getBias().getN() * j.getBias().getM()
              << std::endl;
    sum += (j.getWeight().getM() * j.getWeight().getN() +
            j.getBias().getN() * j.getBias().getM());
    LayerIdx++;
  }
  for (size_t idx = 0; idx < 50; idx++)
    std::cout << "=";
  std::cout << std::endl << "Total Parameters: " << sum << std::endl;
  for (size_t idx = 0; idx < 50; idx++)
    std::cout << "=";
  std::cout << std::endl;
}
void peterzheng::model::Dense::compile() {
  int lastInputSize = this->x.getM();
  // connect the layers
  for (auto &layerConfig : config) {
    this->kernel.push_back(
        DenseCell(matrix::matrix<float>(lastInputSize, 1),
                  matrix::matrix<float>(layerConfig, 1), layerConfig,
                  activationFunction::type::Tanh, matrix::tanh<float>,
                  matrix::tanhGrad<float>));
    lastInputSize = layerConfig;
  }
  this->kernel.push_back(DenseCell(
      matrix::matrix<float>(lastInputSize, 1), matrix::matrix<float>(1, 1), 1,
      activationFunction::type::Sigmoid, matrix::sigmoid<float>,
      matrix::sigmoidGrad<float>));
  for (auto &j : this->kernel) {
    j.compile();
    j.init();
  }
}
float peterzheng::model::Dense::loss() { return 0; }
peterzheng::model::Dense::Dense(
    const peterzheng::matrix::matrix<float> &x,
    const peterzheng::matrix::matrix<float> &y, const int &feature,
    const int &samples, const std::vector<int> &layerConnection,
    float learningRate, peterzheng::model::lossFunction::type lossfunction,
    const std::string &savingPrefix, int savingInterval, int epoches)
    : x(x), y(y), learningRate(learningRate), lossfunction(lossfunction),
      savingPrefix(savingPrefix), savingInterval(savingInterval),
      epoches(epoches), config(layerConnection) {
  if (lossfunction == lossFunction::type::MSE)
    this->lossFunction = matrix::mse;
  if (this->x.getN() != samples && this->x.getM() != samples)
    throw exception(
        "Input x error, should have at least samples columns or samples rows",
        std::string(__FILE__), "InputError", __LINE__);
  if (this->x.getN() != feature && this->x.getM() != feature)
    throw exception(
        "Input x error, should have at least feature columns or samples rows",
        std::string(__FILE__), "InputError", __LINE__);
  if (this->y.getM() != samples && this->y.getN() != samples)
    throw exception(
        "Input x error, should have at least samples columns or samples rows",
        std::string(__FILE__), "InputError", __LINE__);

  if (this->x.getN() == feature)
    this->x.transpose();
  if (this->y.getN() == samples)
    this->y.transpose();
}
