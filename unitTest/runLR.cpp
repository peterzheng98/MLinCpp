//
// Created by 郑文鑫 on 2019/12/10.
//

#include "../linearRegression/logisticRegression.hpp"

int main(){
  int n = 100, m = 32561;
  peterzheng::matrix::matrix<float> x(m, n), y(m, 1);
  std::fstream ifs("1.csv");
  for(size_t idx_m = 0; idx_m < m; ++idx_m) {
    for (size_t idx_n = 0; idx_n < n - 1; ++idx_n) {
      ifs >> x(idx_m, idx_n);
    }
    x(idx_m, n - 1) = 1; // set for bias
    ifs >> y(idx_m, 0);
    assert(y(idx_m, 0) == 0 || y(idx_m, 0) == 1);
  }
  x.dumpToFile("Matrix_x", std::ios::app);
  y.dumpToFile("Matrix_y", std::ios::app);
  peterzheng::model::LogisticRegression lr(x, y, 0.001, "data/training", 1, 1000);
  lr.compile();
  lr.run();
  return 0;
}