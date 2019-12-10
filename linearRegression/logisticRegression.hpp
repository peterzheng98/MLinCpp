//
// Created by 郑文鑫 on 2019/12/2.
//

#ifndef MLINCPP_LOGISTICREGRESSION_H
#define MLINCPP_LOGISTICREGRESSION_H

#include "../matrix/matrix.h"
#include "../matrix/matrixTools.h"
#include "../misc/exception.h"
#include "../misc/usefulTools.h"
#include "../toplevel/model.h"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <random>

namespace peterzheng {
namespace model {
class LogisticRegression : public model {
private:
  peterzheng::matrix::matrix<float> x, y;
  peterzheng::matrix::matrix<float> theta;
  float learningRate;
  std::string savingPrefix;
  int savingInterval;
  int epoches;
  int totalNum;
  static std::random_device r;
  std::default_random_engine randomEngine;
  std::uniform_real_distribution<float> uniformRealDistribution;

public:
  LogisticRegression(const matrix::matrix<float> &x,
                     const matrix::matrix<float> &y, float learningRate = 0.01,
                     const std::string &savingPrefix = "data/training",
                     int savingInterval = 1, int epoches = 100)
      : x(x), y(y), learningRate(learningRate), savingPrefix(savingPrefix),
        savingInterval(savingInterval), epoches(epoches),
        randomEngine(time(0)) {
    this->theta.reload(1, this->x.getN());
    totalNum = x.getM();
  }

  void outputProgress(int currentEpoch, float timePerEpoch) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(4);
    std::cout << "Epoch: " << currentEpoch << "/" << epoches
              << ": Training Loss: " << loss() << " Training Acc: " << acc() * 100 << "%"
              << " Last Epoch: " << timePerEpoch
              << "s ETA: " << timePerEpoch * 1.0 * (epoches - currentEpoch)
              << "s" << std::endl;
  }

public:
  void run() override {
    std::cout << "Start Training: Learning Rate: " << this->learningRate
              << "  Epoches:" << this->epoches << std::endl;

    for (size_t epoch = 0; epoch < epoches; ++epoch) {
      std::chrono::steady_clock::time_point epoch_start =
          std::chrono::steady_clock::now();
      auto sigmoidInternal = x * matrix::getTranspose(theta);
      auto subresult =
          matrix::getTranspose(x) * (y - matrix::sigmoid(sigmoidInternal));
      for (size_t idx = 0; idx < theta.getM(); ++idx)
        theta(idx, 0) = theta(idx, 0) - learningRate * subresult(idx, 0);
      std::chrono::steady_clock::time_point epoch_end =
          std::chrono::steady_clock::now();
      std::chrono::steady_clock::duration epoch_duration =
          epoch_end - epoch_start;
      outputProgress(epoch, std::chrono::duration_cast<std::chrono::seconds>(
                                epoch_duration)
                                .count());
      std::fstream dumpOut(savingPrefix + std::to_string(epoch) +
                               ".parameter.bin",
                           std::ios::out);
      dumpOut << theta.getM() << " " << theta.getN() << std::endl;
      for (size_t idx1 = 0; idx1 < theta.getM(); ++idx1) {
        for (size_t idx2 = 0; idx2 < theta.getN(); ++idx2)
          dumpOut << theta(idx1, idx2) << " ";
        dumpOut << std::endl;
      }
      dumpOut.close();
    }
  }
  void compile() override {
    for (size_t idx = 0; idx < this->theta.getN(); ++idx) {
      theta(0, idx) = uniformRealDistribution(randomEngine);
    }
    theta.dumpToFile("start_theta", std::ios::app);
    return;
  }
  float loss() override {
    float ret = 0.0;
    matrix::matrix<float> sigmoidMat =
        matrix::sigmoid(x * matrix::getTranspose(theta));
    for (size_t idx = 0; idx < sigmoidMat.getM(); ++idx) {
      if (sigmoidMat(idx, 0) < 0)
        throw exception(
            "Sigmoid Function requires the result to be bigger than 0",
            std::string(__FILE__), "ModelError(LR)", __LINE__);
      if (sigmoidMat(idx, 0) > 1)
        throw exception(
            "Sigmoid Function requires the result to be smaller than 1",
            std::string(__FILE__), "ModelError(LR)", __LINE__);
      if (sigmoidMat(idx, 0) == 1)
        sigmoidMat(idx, 0) -= 1e-3;
      if (sigmoidMat(idx, 0) == 0)
        sigmoidMat(idx, 0) += 1e-3;
      auto part_1 = y(idx, 0) * std::log(sigmoidMat(idx, 0));
      auto part_2 = (1 - y(idx, 0)) * (std::log(1 - sigmoidMat(idx, 0)));
      ret += (-part_1 - part_2);
    }
    return ret / 1. / totalNum;
  }

  float acc() {
    float ret = 0.0;
    int count = 0;
    matrix::matrix<float> sigmoidMat =
        matrix::sigmoid(x * matrix::getTranspose(theta));
    for (size_t idx = 0; idx < sigmoidMat.getM(); ++idx)
      if (sigmoidMat(idx, 0) > 0.5 && y(idx, 0) == 1)
        count++;
      else if (sigmoidMat(idx, 0) < 0.5 && y(idx, 0) == 0)
        count++;
    return 1. * count / totalNum;
  }
};
} // namespace model
} // namespace peterzheng

#endif // MLINCPP_LOGISTICREGRESSION_H
