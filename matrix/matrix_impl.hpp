//
// Created by 郑文鑫 on 2019/11/24.
//
#ifndef MLINCPP_MATRIX_IMPL_HPP
#define MLINCPP_MATRIX_IMPL_HPP
#include "exception.h"

namespace peterzheng {
namespace matrix {
template <class T> matrix<T> matrix<T>::operator+(const matrix<T> &rhs) const {
  if (!checkSize<T, T>(*this, rhs))
    throw exception("Matrix size doesn't match[" + genSize(*this) + "],[" +
                        genSize(rhs) + "]",
                    std::string(__FILE__), "ValueError", __LINE__);
  matrix<T> result(this->getM(), this->getN());
  size_t length1 = this->getM();
  size_t length2 = this->getN();
  for (size_t idx1 = 0; idx1 < length1; ++idx1)
    for (size_t idx2 = 0; idx2 < length2; ++idx2)
      result(idx1, idx2) = this->operator()(idx1, idx2) + rhs(idx1, idx2);
  return result;
}

template <class T> matrix<T> matrix<T>::operator-(const matrix<T> &rhs) const {
  if (!checkSize<T, T>(*this, rhs))
    throw exception("Matrix size doesn't match[" + genSize(*this) + "],[" +
                        genSize(rhs) + "]",
                    std::string(__FILE__), "ValueError", __LINE__);
  matrix<T> result(this->getM(), this->getN());
  size_t length1 = this->getM();
  size_t length2 = this->getN();
  for (size_t idx1 = 0; idx1 < length1; ++idx1)
    for (size_t idx2 = 0; idx2 < length2; ++idx2)
      result(idx1, idx2) = this->operator()(idx1, idx2) - rhs(idx1, idx2);
  return result;
}

template <class T> matrix<T> matrix<T>::operator*(const matrix<T> &rhs) const {
  size_t thisn = this->getN(), thism = this->getM(), rn = rhs.getN(),
         rm = rhs.getM();
  if (thisn != rm)
    throw exception("Matrix size doesn't match[" + genSize(*this) + "],[" +
                        genSize(rhs) + "]",
                    std::string(__FILE__), "ValueError", __LINE__);
  matrix<T> result(thism, rn);
  for (size_t idx1 = 0; idx1 < thism; ++idx1)
    for (size_t idx2 = 0; idx2 < rn; ++idx2) {
      for (size_t idx3 = 0; idx3 < thisn; ++idx3) {
        result(idx1, idx2) += this->operator()(idx1, idx3) * rhs(idx3, idx2);
      }
    }
  return result;
}

template <class T> matrix<T> &matrix<T>::operator+=(const matrix<T> &rhs) {
  this = this + rhs;
  return this;
}

template <class T> matrix<T> &matrix<T>::operator-=(const matrix<T> &rhs) {
  this = this - rhs;
  return this;
}

template <class T> matrix<T> &matrix<T>::operator*=(const matrix<T> &rhs) {
  this = this * rhs;
  return this;
}
template <class U, class T>
matrix<decltype(U() * T())> operator*(const matrix<T> &lhs, const U &rhs) {
  matrix<decltype(U() * T())> result(lhs.getM(), lhs.getN());
  size_t length1 = lhs.getM();
  size_t length2 = lhs.getN();
  for (size_t idx1 = 0; idx1 < length1; ++idx1)
    for (size_t idx2 = 0; idx2 < length2; ++idx2)
      result(idx1, idx2) = lhs(idx1, idx2) * rhs;
  return result;
}

template <class T, class U>
matrix<decltype(T() * U())> operator*(const matrix<T> &lhs,
                                      const matrix<U> &rhs) {
  size_t thisn = lhs->getN(), thism = lhs->getM(), rn = rhs.getN(),
         rm = rhs.getM();
  if (thisn != rm)
    throw exception("Matrix size doesn't match[" + genSize(lhs) + "],[" +
                        genSize(rhs) + "]",
                    std::string(__FILE__), "ValueError", __LINE__);
  matrix<T> result(thism, rn);
  for (size_t idx1 = 0; idx1 < thism; ++idx1)
    for (size_t idx2 = 0; idx2 < rn; ++idx2) {
      for (size_t idx3 = 0; idx3 < thisn; ++idx3) {
        result(idx1, idx2) += lhs(idx1, idx3) * rhs(idx3, idx2);
      }
    }
  return result;
}

template <class T>
template <class U>
matrix<decltype(U() * T())> &matrix<T>::operator*=(const U &rhs) {
  this = this * rhs;
  return this;
}
} // namespace matrix

}
#endif