//
// Created by 郑文鑫 on 2019/12/20.
//

#include "Dense.h"
#include "../matrix/matrixTools.h"
#include <cmath>
template <class T>
T abs(const T& a){ return a > 0 ? a : -a; }
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
peterzheng::matrix::matrix<float> peterzheng::model::DenseCell::run(
    const peterzheng::matrix::matrix<float> &x) {
  this->input = x;
  if(this->activation == activationFunction::Tanh)
    this->output = matrix::tanh(weight * x + bias);
  else if(this->activation == activationFunction::Sigmoid)
    this->output = matrix::sigmoid(weight * x + bias);
  return this->output;
}

peterzheng::matrix::matrix<float> peterzheng::model::DenseCell::eval(
    const peterzheng::matrix::matrix<float> &x) {
  return this->activation == activationFunction::Tanh ? matrix::tanh(weight * x + bias) : (this->activation == activationFunction::Sigmoid ? this->output = matrix::sigmoid(weight * x + bias): this->output = matrix::sigmoid(weight * x + bias));
}

void peterzheng::model::DenseCell::compile() {
  srand(time(0));
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
      weight(idx1, idx2) = rand() / float(RAND_MAX);
    }
    bias(idx1, 0) = rand() / float(RAND_MAX);
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
const std::function<peterzheng::matrix::matrix<float>(peterzheng::matrix::matrix<float>)> &
peterzheng::model::DenseCell::getActivationGrad() const {
  return activationGrad;
}
void peterzheng::model::DenseCell::setWeight(
    const peterzheng::matrix::matrix<float> &weight) {
  DenseCell::weight = weight;
}
const std::function<peterzheng::matrix::matrix<float>(peterzheng::matrix::matrix<float>)> &
peterzheng::model::DenseCell::getActivationfunc() const {
  return activationfunc;
}
peterzheng::model::activationFunction::type
peterzheng::model::DenseCell::getActivation() const {
  return activation;
}
// Basic Idea:

void peterzheng::model::Dense::run() {
  // TODO
  std::cout << "Start Training with learning rate at:" << this->learningRate << " Epoches:" << this->epoches << " Batch:" << this->batchSize << std::endl;
  for(size_t currentEpoch = 0; currentEpoch < this->epoches; ++currentEpoch){
    // Loading timing part
    std::chrono::steady_clock::time_point epoch_start = std::chrono::steady_clock::now();

    // select data
    int slice = totalNum / batchSize;
    int lower, higher;
    for(int batchRound = 0; batchRound <= slice; ++batchRound){
      //Timing for batch
      std::chrono::steady_clock::time_point slice_start = std::chrono::steady_clock::now();

      lower = batchRound * batchSize;
      higher = std::min((batchRound + 1) * batchSize - 1, totalNum - 1);
      std::uniform_int_distribution<unsigned>::param_type param(lower, higher);
      uniformIntDistribution.param(param);
      int target = abs<int>(rand() % (higher - lower)) + lower;

      // Calculate forward
      auto lastInput = matrix::getColumn(x, target);
      auto basicStandard = matrix::getRow(y, target);
      for(auto &layer : this->kernel){
        lastInput = layer.run(lastInput);
      }
      // Calculate backward and update
      // Step 1: Calculate errors
      matrix::matrix<float> ErrorETA(1, y.getN());
      for(size_t idx = 0; idx < y.getN(); ++idx)
        ErrorETA(0, idx) = 1.0 * (lastInput(0, idx) - basicStandard(0, idx)) * (1 - lastInput(0, idx)) * lastInput(0, idx);
      // Step 2: Update the weight and back prop
      size_t length = this->kernel.size();
      for(size_t LayerIdx = length - 1; LayerIdx > 0; --LayerIdx){
        auto oldOutput = this->kernel[LayerIdx].getOutput();
        auto oldInput = this->kernel[LayerIdx].getInput();
        auto beforeWeight = this->kernel[LayerIdx].getWeight();
        auto newWeight = beforeWeight + matrix::getTranspose((oldInput * ErrorETA)) * learningRate;
        ErrorETA = ErrorETA * beforeWeight;
        auto type = this->kernel[LayerIdx].getActivation();
        for(size_t idx = 0; idx < ErrorETA.getN(); ++idx){
          if(type == activationFunction::Sigmoid) ErrorETA(0, idx) = ErrorETA(0, idx) * matrix::sigmoidGrad(oldInput)(idx, 0);
          else if(type == activationFunction::Tanh) ErrorETA(0, idx) = ErrorETA(0, idx) * matrix::tanhGrad(oldInput)(idx, 0);
        }

        this->kernel[LayerIdx].setWeight(newWeight);
      }
      std::chrono::steady_clock::time_point slice_end = std::chrono::steady_clock::now();
      std::chrono::steady_clock::duration slice_duration = slice_end - slice_start;
      outputProgressBarFlush(currentEpoch, batchRound, slice, std::chrono::duration_cast<std::chrono::milliseconds>(slice_duration).count() / 1000.0);
    }
    std::chrono::steady_clock::time_point epoch_end =
        std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration epoch_duration =
        epoch_end - epoch_start;
    outputProgress(
        currentEpoch,
        std::chrono::duration_cast<std::chrono::milliseconds>(epoch_duration).count() / 1000.0);
  }

}

void peterzheng::model::Dense::outputProgressBarFlush(int currentEpoch, int slice, int sliceAll,
                            float timePerSlice) {
  std::cout << "Epoch: " << currentEpoch << "/" << this->epoches
            << ": Slice Progress: " << slice * 100.0 / sliceAll
            // << " == Training Loss: " << loss()
            << " Last Slice: " << timePerSlice << "s ETA: "
            << timePerSlice * 1.0 * (sliceAll - slice) + timePerSlice * 1.0 * sliceAll *
               (this->epoches - currentEpoch - 1)
            << "s\r" << std::flush;
}
void peterzheng::model::Dense::outputProgress(int currentEpoch, float timePerEpoch) {
  auto lossR = loss();
  for(size_t idx = 0; idx < 50; ++idx) std::cout << " ";
  std::cout << "\r" << std::flush;
  std::cout << "Epoch: " << currentEpoch << "/" << this->epoches
            << ": Training Loss: " << lossR
            << " Last Epoch: " << timePerEpoch
            << "s ETA: " << timePerEpoch * 1.0 * (this->epoches - currentEpoch)
            << "s" << std::endl;
}
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
    std::cout << "Dense_" << LayerIdx << "\t\t\t Input Size:(" << j.getInput().getM() << "," << j.getInput().getN() << ") <--Weight:(" << j.getWeight().getM()
              << "," << j.getWeight().getN() << ")--> Output Size:(" << j.getOutput().getM() << "," << j.getOutput().getN() << ")\t\t"
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
float peterzheng::model::Dense::loss() {
  std::cout << "Evaluate on all data" << std::endl;
  int ACC = 0;
  double lossSum = 0.0;
  for(size_t target = 0; target < x.getN(); ++target) {
    auto lastInput = matrix::getColumn(x, target);
    auto basicStandard = matrix::getRow(y, target);
    for (auto &layer : this->kernel) {
      lastInput = layer.eval(lastInput);
    }
    lossSum += ((lastInput(0, 0) - basicStandard(0, 0)) * (lastInput(0, 0) - basicStandard(0, 0)));
    if(lastInput(0, 0) >= 0.5 && basicStandard(0,0)) ACC++;
    else if(lastInput(0, 0) < 0.5 && !basicStandard(0, 0)) ACC++;
    std::cout << "\t  LastInput:" << lastInput(0, 0) << "->" << basicStandard(0, 0) << "\t";
    std::cout << "\t Target: " << target << " Remaining: " << x.getN() - target << " loss: " << lossSum << "\r" << std::flush;
  }
  std::cout << "Acc: " <<  100.0 * ACC / totalNum << "\t";
  return lossSum / totalNum / 2.0;
}
peterzheng::model::Dense::Dense(
    const peterzheng::matrix::matrix<float> &x,
    const peterzheng::matrix::matrix<float> &y, const int &feature,
    const int &samples, const std::vector<int> &layerConnection,
    float learningRate, peterzheng::model::lossFunction::type lossfunction,
    const std::string &savingPrefix, int savingInterval, int epoches, int batchSize)
    : x(x), y(y), learningRate(learningRate), lossfunction(lossfunction),
      savingPrefix(savingPrefix), savingInterval(savingInterval),
      epoches(epoches), config(layerConnection), batchSize(batchSize) {
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
        "Input y error, should have at least samples columns or samples rows",
        std::string(__FILE__), "InputError", __LINE__);

  if (this->x.getN() == feature)
    this->x.transpose();
  if (this->y.getN() == samples)
    this->y.transpose();
  totalNum = samples;
}
