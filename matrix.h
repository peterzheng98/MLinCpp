//
// Created by 郑文鑫 on 2019/11/24.
//

#ifndef MLINCPP_MATRIX_H
#define MLINCPP_MATRIX_H
#include <string>
#include <vector>
#include "misc/usefulTools.h"
namespace peterzheng {
/*
 * Class matrix:
 * give out a matrix class that solves two-dimension matrix calculation
 */
template <class T> class matrix {
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
  matrix(const matrix<T> &rhs) {
    m = rhs.getM();
    n = rhs.getN();
    data.resize(m);
    for (size_t idx = 0; idx < m; ++idx) {
      data[idx].resize(n);
      for (size_t idx2 = 0; idx2 < n; ++idx2) {
        data[idx][idx2] = rhs(idx, idx2);
      }
    }
  }

  matrix&operator=(const matrix<T>& rhs){
    if(this == rhs) return this;
    m = rhs.getM();
    n = rhs.getN();
    data.resize(m);
    for (size_t idx = 0; idx < m; ++idx) {
      data[idx].resize(n);
      for (size_t idx2 = 0; idx2 < n; ++idx2) {
        data[idx][idx2] = rhs(idx, idx2);
      }
    }
  }

  matrix(size_t m, size_t n) : m(m), n(n) {
    data.resize(m);
    for (auto &elem : data)
      elem.resize(n);
  }
  matrix<T> operator+(const matrix<T> &rhs);
  matrix<T> operator-(const matrix<T> &rhs);
  matrix<T> operator*(const matrix<T> &rhs);
  matrix<T> &operator+=(const matrix<T> &rhs);
  matrix<T> &operator-=(const matrix<T> &rhs);
  matrix<T> &operator*=(const matrix<T> &rhs);
  size_t getM() const { return m; }
  void setM(size_t m) { matrix::m = m; }
  size_t getN() const { return n; }
  void setN(size_t n) { matrix::n = n; }
  T &operator()(const int idx1, const int idx2) { return data[idx1][idx2]; }

  T &operator()(const size_t idx1, const size_t idx2) {
    return data[idx1][idx2];
  }

public:
  template <class U> matrix<decltype(U() * T())> operator*(const U &rhs);
  template <class U> matrix<decltype(U() * T())> &operator*=(const U &rhs);

};


template <class T, class U> bool checkSize(matrix<T> &r1, matrix<U> &r2) {
  return r1.getM() == r2.getM() && r1.getN() == r2.getN();
}
std::string genSize(matrix<T> &r1) {
  return std::to_string(r1.getM()) + "," + std::to_string(r1.getN());
}
template <class T>
matrix<T> getIdentity(size_t t){
  matrix<T> result(t, t);
  for(size_t idx = 0; idx < t; ++idx)
    result(idx, idx) = T(1);
  return result;
}
} // namespace peterzheng

#endif // MLINCPP_MATRIX_H
