//
// Created by 郑文鑫 on 2019/12/3.
//

#ifndef MLINCPP_LINEARREGRESSION_HPP
#define MLINCPP_LINEARREGRESSION_HPP
#include "../matrix/matrix.h"
#include "../misc/exception.h"
#include "../toplevel/model.h"
#include <algorithm>
#include <chrono>
#include <random>
namespace peterzheng {
namespace model {
enum GradientDescentPolicy { BatchGradientDescent, StochasticGradientDescent };

class LinearRegression : public model {
private:
  peterzheng::matrix::matrix<float> x, y, theta;
  GradientDescentPolicy policy;
  float learningRate;
  std::string savingPrefix;
  int savingInterval;
  int epochs;
  int batchSize;
  static std::default_random_engine randomEngine;
  static std::uniform_real_distribution<float> uniformRealDistribution;
  static std::uniform_int_distribution<unsigned> uniformIntDistribution;

public:
  LinearRegression(const matrix::matrix<float> &rx,
                   const matrix::matrix<float> &ry,
                   const GradientDescentPolicy &policy1,
                   const int &targetEpoches, const float &lr = 0.01,
                   const int &batch = 128,
                   const std::string &savingPrefix = "data/training",
                   const int &savingInterval = 1)
      : x(rx), y(ry), policy(policy1), learningRate(lr),
        savingPrefix(savingPrefix), savingInterval(savingInterval),
        epochs(targetEpoches), batchSize(batch) {
    if (y.getM() == 1)
      y.transpose();
    else if (y.getM() != 1 && y.getN() != 1)
      throw exception("Label Matrix should be in the form of vector.",
                      std::string(__FILE__), "ValueError", __LINE__);
    theta.reload(y.getM(), 1);
  }

private:
  float loss() override {
    float loss = 0;
    int N = theta.getM();
    matrix::matrix<float> caledResult = y - x * theta;
    if (caledResult.getN() != 1)
      throw exception("Size of parameter theta error! Expect the column size "
                      "to be one, real is " +
                          std::to_string(caledResult.getN()),
                      std::string(__FILE__), "InternalError", __LINE__);
    // the size of caledResult should be n x 1
    for (size_t idx = 0; idx < caledResult.getM(); ++idx) {
      loss += (caledResult(idx, 0) * caledResult(idx, 0));
    }
    loss /= (N << 1);
    return loss;
  }

  float calculateYi(const size_t &tar) const {
    float sum = 0;
    for (size_t idx = 0; idx < x.getN(); ++idx)
      sum += (x(tar, idx) * theta(idx, 1));
    return y(tar, 1) - sum;
  }

  void outputProgressBarFlush(int currentEpoch, int slice, int sliceAll,
                              float timePerSlice) {
    std::cout << "Epoch: " << currentEpoch << "/" << epochs
              << ": Slice Progress: " << slice * 100.0 / sliceAll
              << " == Training Loss: " << loss()
              << " Last Slice: " << timePerSlice << "s ETA: "
              << timePerSlice * 1.0 * (sliceAll - slice) *
                     (epochs - currentEpoch)
              << "s\r" << std::flush;
  }
  void outputProgress(int currentEpoch, float timePerEpoch) {
    std::cout << "Epoch: " << currentEpoch << "/" << epochs
              << ": Training Loss: " << loss()
              << " Last Epoch: " << timePerEpoch
              << "s ETA: " << timePerEpoch * 1.0 * (epochs - currentEpoch)
              << "s" << std::endl;
  }

public:
  void run() override {
    std::cout << "Starting Training: "
              << (policy == GradientDescentPolicy::BatchGradientDescent
                      ? "Batch Gradient Descent"
                      : "Stochastic Gradient Descent")
              << " Learning Rate: " << learningRate << " Epochs: " << epochs
              << " Batch Size: " << batchSize << std::endl;
    int currentEpoch = 1;
    int totalN = theta.getM();
    while (currentEpoch < epochs) {
      std::chrono::steady_clock::time_point epoch_start =
          std::chrono::steady_clock::now();
      if (policy == GradientDescentPolicy::StochasticGradientDescent) {
        int slice = totalN / batchSize;
        int lower, higher;
        for (int round = 0; round <= slice; ++round) {
          std::chrono::steady_clock::time_point slice_start =
              std::chrono::steady_clock::now();
          lower = round * batchSize;
          higher = std::min(round * batchSize - 1, totalN - 1);
          std::uniform_int_distribution<unsigned>::param_type param(lower,
                                                                    higher);
          uniformIntDistribution.param(param);
          int target = uniformIntDistribution(randomEngine);
          float realY = y(target, 0);
          float expectedY = 0;
          for (size_t idx = 0; idx < totalN; ++idx)
            expectedY += theta(idx, 0) * x(target, idx);
          float diff = realY - expectedY;
          for (size_t idx = 0; idx < theta.getM(); ++idx)
            theta(idx, 0) = theta(idx, 0) +
                            learningRate * calculateYi(target) * x(target, idx);
          std::chrono::steady_clock::time_point slice_end =
              std::chrono::steady_clock::now();
          std::chrono::steady_clock::duration slice_duration =
              slice_end - slice_start;
          outputProgressBarFlush(
              currentEpoch, round, slice,
              std::chrono::duration_cast<std::chrono::seconds>(slice_duration)
                  .count());
        }
        std::chrono::steady_clock::time_point epoch_end =
            std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration epoch_duration =
            epoch_end - epoch_start;
        outputProgress(
            currentEpoch,
            std::chrono::duration_cast<std::chrono::seconds>(epoch_duration)
                .count());
      } else { // policy is BGD
        int slice = totalN / batchSize;
        int lower, higher;
        for (int round = 0; round <= slice; ++round) {
          std::chrono::steady_clock::time_point slice_start =
              std::chrono::steady_clock::now();
          lower = round * batchSize;
          higher = std::min(round * batchSize - 1, totalN - 1);
          matrix::matrix<float> tempDiff(theta.getM(), theta.getN());
          for (size_t idx1 = 0; idx1 < theta.getM(); ++idx1)
            for (size_t idx2 = 0; idx2 < theta.getN(); ++idx2)
              tempDiff(idx1, idx2) = 0;
          for (size_t target = lower; target <= higher; ++target) {
            for (size_t idx = 0; idx < theta.getM(); ++idx) {
              tempDiff(idx, 0) += calculateYi(target) * x(target, idx);
            }
          }
          for (size_t idx = 0; idx < theta.getM(); ++idx)
            theta(idx, 0) = theta(idx, 0) +
                            learningRate * (1.0 / batchSize) * tempDiff(idx, 0);
          std::chrono::steady_clock::time_point slice_end =
              std::chrono::steady_clock::now();
          std::chrono::steady_clock::duration slice_duration =
              slice_end - slice_start;
          outputProgressBarFlush(
              currentEpoch, round, slice,
              std::chrono::duration_cast<std::chrono::seconds>(slice_duration)
                  .count());
        }
        std::chrono::steady_clock::time_point epoch_end =
            std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration epoch_duration =
            epoch_end - epoch_start;
        outputProgress(
            currentEpoch,
            std::chrono::duration_cast<std::chrono::seconds>(epoch_duration)
                .count());
      }
      currentEpoch++;
      std::fstream dumpOut(savingPrefix + std::to_string(currentEpoch) + ".parameter.bin", std::ios::out);
      dumpOut << theta.getM() << " " << theta.getN() << std::endl;
      for(size_t idx1 = 0; idx1 < theta.getM(); ++idx1) {
        for (size_t idx2 = 0; idx2 < theta.getN(); ++idx2)
          dumpOut << theta(idx1, idx2) << " ";
        dumpOut << std::endl;
      }
      dumpOut.close();
    }
  }
  void compile() override {
    for (size_t idx = 0; idx < this->theta.getM(); ++idx) {
      theta(idx, 0) = uniformRealDistribution(randomEngine);
    }
  }
};
} // namespace model
} // namespace peterzheng

#endif