//
// Created by 郑文鑫 on 2019/12/3.
//

#ifndef MLINCPP_LINEARREGRESSION_HPP
#define MLINCPP_LINEARREGRESSION_HPP
#include <random>
#include <algorithm>
#include "../matrix/matrix.h"
#include "../misc/exception.h"
#include "../toplevel/model.h"
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

public:
  void run() override {
    std::cout << "Starting Training: "
              << (policy == GradientDescentPolicy::BatchGradientDescent
                      ? "Batch Gradient Descent"
                      : "Stochastic Gradient Descent")
              << " Learning Rate: " << learningRate << " Epochs: " << epochs
              << " Batch Size: " << batchSize << std::endl;
    int currentEpoch = 1;
//    if(policy == GradientDescentPolicy::BatchGradientDescent) batchSize = 1;
    int totalN = theta.getM();
    while(currentEpoch < epochs){
      if(policy == GradientDescentPolicy::StochasticGradientDescent){
        int slice = totalN / batchSize;
        int lower, higher;
        for(int round = 0; round <= slice; ++round){
          lower = round * batchSize;
          higher = std::min(round * batchSize - 1, totalN - 1);
          std::uniform_int_distribution<unsigned>::param_type param(lower, higher);
          uniformIntDistribution.param(param);
          int target = uniformIntDistribution(randomEngine);
          float realY = y(target, 0);
          float expectedY = 0;
          for(size_t idx = 0; idx < totalN; ++idx)
            expectedY += theta(idx, 0) * x(target, idx);
          float diff = realY - expectedY;
          theta = theta + learningRate * (x);

        }
      }
      float currentloss = loss();

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