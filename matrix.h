//
// Created by 郑文鑫 on 2019/11/24.
//

#ifndef MLINCPP_MATRIX_H
#define MLINCPP_MATRIX_H
#include <vector>
namespace peterzheng {
/*
 * Class matrix:
 * give out a matrix class that solves two-dimension matrix calculation
 */
template <class T>
class matrix {
private:
  std::vector<std::vector<T>> data;
  size_t m, n; // indicates m rows with n column matrix

public:
  matrix() {
    m = n = 10;
    data.resize(10);
    for (auto &elem : data)
      elem.resize(10);
  }

  matrix(const std::vector<std::vector<T>> &data, size_t m, size_t n)
      : data(data), m(m), n(n) {}

  matrix(size_t m, size_t n) : m(m), n(n) {
    data.resize(m);
    for(auto &elem: data) elem.resize(n);
  }
  matrix<T> operator+(const matrix& rhs){
    matrix<T> ret(m, n);

  }
  size_t getM() const { return m; }
  void setM(size_t m) { matrix::m = m; }
  size_t getN() const { return n; }
  void setN(size_t n) { matrix::n = n; }
};
} // namespace peterzheng

#endif // MLINCPP_MATRIX_H
