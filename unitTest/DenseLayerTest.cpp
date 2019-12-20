//
// Created by 郑文鑫 on 2019/12/20.
//

#include "../neuralNetworks/Dense.h"
#include "../matrix/matrix.h"
#include <cmath>
//#include "../neuralNetworks/parameter.h"
peterzheng::model::Dense* core = nullptr;
int main(){
  std::vector<int> rtt;
  for(int i = 5; i < 10; ++i) rtt.push_back(int(std::pow(2, i)));
  for(int i = 10; i > 1; --i) rtt.push_back(int(std::pow(2, i)));
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
  core = new peterzheng::model::Dense(x, y, n, m, rtt);
  core->compile();
  core->summary();
  return 0;

}