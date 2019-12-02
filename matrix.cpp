//
// Created by 郑文鑫 on 2019/11/24.
//

#include "matrix.h"
#include "misc/exception.h"

namespace peterzheng {
template <class T> matrix<T> matrix<T>::operator+(const matrix<T> &rhs) {
  if (!checkSize(this, rhs))
    throw exception("Matrix size doesn't match[" + genSize(this) + "],[" +
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

template <class T> matrix<T> matrix<T>::operator-(const matrix<T> &rhs) {
  if (!checkSize(this, rhs))
    throw exception("Matrix size doesn't match[" + genSize(this) + "],[" +
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

template <class T> matrix<T> matrix<T>::operator*(const matrix<T> &rhs) {
  size_t thisn = this->getN(), thism = this->getM(), rn = rhs.getN(),
         rm = rhs.getM();
  if (thisn != rm)
    throw exception("Matrix size doesn't match[" + genSize(this) + "],[" +
                        genSize(rhs) + "]",
                    std::string(__FILE__), "ValueError", __LINE__);
  matrix<T> result(thism, rn);
  for(size_t idx1 = 0; idx1 < thism; ++idx1)
    for(size_t idx2 = 0; idx2 < rn; ++idx2){
      for(size_t idx3 = 0; idx3 < thisn; ++idx3){
        result(idx1, idx2) += this->operator()(idx1, idx3) * rhs(idx3, idx2);
      }
    }
  return result;
}

template <class T> matrix<T> &matrix<T>::operator+=(const matrix<T>& rhs){
  this = this + rhs;
  return this;
}

template <class T> matrix<T> &matrix<T>::operator-=(const matrix<T>& rhs){
  this = this - rhs;
  return this;
}

template <class T> matrix<T> &matrix<T>::operator*=(const matrix<T>& rhs){
  this = this * rhs;
  return this;
}
template <class T>
template <class U>
matrix<decltype(U() * T())> matrix<T>::operator*(const U &rhs) {
  matrix<decltype(U() * T())> result(this->getM(), this->getN());
  size_t length1 = this->getM();
  size_t length2 = this->getN();
  for (size_t idx1 = 0; idx1 < length1; ++idx1)
    for (size_t idx2 = 0; idx2 < length2; ++idx2)
      result(idx1, idx2) = this->operator()(idx1, idx2) * rhs;
  return result;
}

template <class T>
template <class U>
matrix<decltype(U() * T())> matrix<T>::operator*=(const U &rhs) {
  this = this * rhs;
  return this;
}

}