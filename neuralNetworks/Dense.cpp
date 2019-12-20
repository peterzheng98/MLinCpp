//
// Created by 郑文鑫 on 2019/12/20.
//

#include "Dense.h"
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
  if(input.getM() == 1) input.transpose();
  // hidden_output = f(Wx+b)
  // W: outputSize * m
  // x: m * 1             ----> outputSize * 1
  // b: outputSize * 1

}
float peterzheng::model::DenseCell::loss() {
  throw exception("No loss available for dense cell", std::string(__FILE__), "Internal Error", __LINE__);
}

void peterzheng::model::DenseCell::init(){
  weight.reload(outputSize, input.getM());
  bias.reload(outputSize, 1);
  output.reload(outputSize, 1); // Hidden States
  for(size_t idx1 = 0; idx1 < outputSize; ++idx1){
    for(size_t idx2 = 0; idx2 < input.getM(); ++idx2){
      weight(idx1, idx2) = uniformRealDistribution(randomEngine);
    }
    bias(idx1, 0) = uniformRealDistribution(randomEngine);
    output(idx1, 0) = 0.0;
  }
}

void peterzheng::model::DenseCell::update(const float& eta, const matrix::matrix<float>& grad) {
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
void peterzheng::model::DenseCell::setOutput(
    const peterzheng::matrix::matrix<float> &output) {
  DenseCell::output = output;
}


peterzheng::model::Dense::Dense(
    const peterzheng::matrix::matrix<float> &x,
    const peterzheng::matrix::matrix<float> &y,
    const int& feature, const int& samples, const std::vector<int>& layerConnection,
    float learningRate = 0.1,
    peterzheng::model::lossFunction::type lossfunction = lossFunction::type::MSE,
    const std::function<matrix::matrix<float>(
        matrix::matrix<float>, matrix::matrix<float>)> &lossFunction = lossFunction::mse,
    const std::string &savingPrefix = "data/training", int savingInterval = 1, int epoches = 128)
    : x(x), y(y), learningRate(learningRate), lossfunction(lossfunction),
      lossFunction(lossFunction), savingPrefix(savingPrefix),
      savingInterval(savingInterval), epoches(epoches) {
  if(this->x.getN() != samples && this->x.getM() != samples) throw exception("Input x error, should have at least samples columns or samples rows", std::string(__FILE__), "InputError", __LINE__);
  if(this->x.getN() != feature && this->x.getM() != feature) throw exception("Input x error, should have at least feature columns or samples rows", std::string(__FILE__), "InputError", __LINE__);
  if(this->y.getM() != samples && this->y.getN() != samples) throw exception("Input x error, should have at least samples columns or samples rows", std::string(__FILE__), "InputError", __LINE__);

  if(this->x.getN() == feature) this->x.transpose();
  if(this->y.getN() == samples) this->y.transpose();
  // convert layer connection to the real layer.

}
