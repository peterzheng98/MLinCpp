//
// Created by 郑文鑫 on 2019/11/24.
//

#ifndef MLINCPP_MATRIX_H
#define MLINCPP_MATRIX_H
#include "../misc/exception.h"
#include "../misc/usefulTools.h"
#include "matrixDef.h"
#include <fstream>
#include <string>
#include <vector>
namespace peterzheng {
namespace matrix {
/*
 * Class matrix:
 * give out a matrix class that solves two-dimension matrix calculation
 */
template <class T> class matrix {
private:
  std::vector<std::vector<T>> data;
  size_t m, n; // indicates m rows with n column matrix
  NORM matrix_cmp = matrix_1;

public:
  matrix() {
    m = n = 10;
    data.resize(10);
    for (auto &elem : data)
      elem.resize(10);
  }

  void reload(size_t m, size_t n){
    this->m = m;
    this->n = n;
    data.resize(m);
    for (auto &elem : data)
      elem.resize(n);
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
        data[idx][idx2] = rhs.data[idx][idx2];
      }
    }
  }

  matrix<T> &operator=(const matrix<T> &rhs) {
    if (this == &rhs)
      return *this;
    m = rhs.getM();
    n = rhs.getN();
    data.resize(m);
    for (size_t idx = 0; idx < m; ++idx) {
      data[idx].resize(n);
      for (size_t idx2 = 0; idx2 < n; ++idx2) {
        data[idx][idx2] = rhs(idx, idx2);
      }
    }
    return *this;
  }
  template <class U>
  matrix<decltype(T() + U())> &operator=(const matrix<U> &rhs) {
    if (this == &rhs)
      return *this;
    m = rhs.getM();
    n = rhs.getN();
    data.resize(m);
    for (size_t idx = 0; idx < m; ++idx) {
      data[idx].resize(n);
      for (size_t idx2 = 0; idx2 < n; ++idx2) {
        data[idx][idx2] = rhs(idx, idx2);
      }
    }
    return *this;
  }

  matrix(size_t m, size_t n) : m(m), n(n) {
    data.resize(m);
    for (auto &elem : data)
      elem.resize(n);
  }
  matrix<T> operator+(const matrix<T> &rhs) const;
  matrix<T> operator-(const matrix<T> &rhs) const;
  matrix<T> operator*(const matrix<T> &rhs) const;
  matrix<T> &operator+=(const matrix<T> &rhs);
  matrix<T> &operator-=(const matrix<T> &rhs);
  matrix<T> &operator*=(const matrix<T> &rhs);
  size_t getM() const { return m; }
  void setM(size_t m) { matrix::m = m; }
  size_t getN() const { return n; }
  void setN(size_t n) { matrix::n = n; }
  T &operator()(const int idx1, const int idx2) { return data[idx1][idx2]; }
  const T &operator()(const int idx1, const int idx2) const {
    return data[idx1][idx2];
  }
  void dumpToFile(const std::string &name,
                  const std::ios_base::openmode &mode) {
    std::fstream f1(name, mode);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j)
        f1 << data[i][j] << " ";
      f1 << std::endl;
    }
    f1 << std::endl;
    f1.close();
  }
  void transpose(){
    matrix<T> ret(n, m);
    for(size_t row = 0; row < this->getN(); ++row)
      for(size_t col = 0; col < this->getM(); ++col)
        ret(row, col) = data[col][row];
    *this = ret;
  }

  void minorMatrix(const size_t &d){
    for(size_t idx = 0; idx < d; ++idx)
      for(size_t idx2 = 0; idx2 < d; ++idx2)
        data[idx][idx2] = (idx == idx2) ? T(1) : T(0);
  }

  bool operator==(const matrix<T>& rhs){
    if(rhs.getN() != n) return false;
    if(rhs.getM() != m) return false;
    for(size_t row = 0; row < m; ++row)
      for(size_t col = 0; col < n; ++col)
        if(rhs(row, col) != data[row][col]) return false;
    return true;
  }
  bool operator!=(const matrix<T> &rhs){
    return !operator==(rhs);
  }

public:
  template <class U> matrix<decltype(U() * T())> &operator*=(const U &rhs);
};

template <class T, class U>
bool checkSize(const matrix<T> &r1, const matrix<U> &r2) {
  return r1.getM() == r2.getM() && r1.getN() == r2.getN();
}

template <class T, class U> bool checkSize(matrix<T> &r1, const matrix<U> &r2) {
  return r1.getM() == r2.getM() && r1.getN() == r2.getN();
}

template <class T> bool checkSize(const matrix<T> &r1, const matrix<T> &r2) {
  return r1.getM() == r2.getM() && r1.getN() == r2.getN();
}
template <class T> std::string genSize(const matrix<T> &r1) {
  return std::to_string(r1.getM()) + "," + std::to_string(r1.getN());
}
template <class T> matrix<T> getIdentity(size_t t) {
  matrix<T> result(t, t);
  for (size_t idx = 0; idx < t; ++idx)
    result(idx, idx) = T(1);
  return result;
}
template <class U, class T>
matrix<decltype(U() * T())> operator*(const matrix<T> &lhs, const U &rhs);
template <class T, class U>
matrix<decltype(T() * U())> operator*(const matrix<T> &lhs,
                                      const matrix<U> &rhs);
}
} // namespace peterzheng

#include "matrix_impl.hpp"
#endif // MLINCPP_MATRIX_H
