//
// Created by 郑文鑫 on 2019/12/3.
//

#ifndef MLINCPP_MATRIXTOOLS_H
#define MLINCPP_MATRIXTOOLS_H

#include "../misc/exception.h"
#include "../misc/usefulTools.h"
#include "matrix.h"
#include "matrixDef.h"
#include <cmath>
#include <iostream>
namespace peterzheng {
namespace matrix {
template <class T>
std::vector<float> eigen(const matrix<T> &mat, matrix<float> &V,
                         matrix<float> &U) {
  // Todo: Eigen Value Decomposition
  return std::vector<float>();
}

template <class T>
std::vector<float> svd(const matrix<T> &mat, matrix<float> &V,
                       matrix<float> &U) {
  // Todo: Singular Value Decomposition
  return std::vector<float>();
}

template <class T> float norm(const matrix<T> &mat, NORM target, int p = 0) {
  if (target == vector_1 || target == vector_2 || target == vector_infty ||
      target == vector_p) {
    if (mat.getM() != 1 && mat.getN() != 1)
      throw exception(
          "Vector Norm can only applied to vector object, current mat size:" +
              std::to_string(mat.getM()) + ", " + std::to_string(mat.getN()),
          std::string(__FILE__), "ValueError", __LINE__);
    if (target == vector_p && p <= 2)
      throw exception("P norm should specify a p value with bigger than 2.",
                      std::string(__FILE__), "ValueError", __LINE__);
    if (target == vector_infty) {
      float ret = mat(0, 0);
      for (size_t dimx = 0; dimx < mat.getM(); ++dimx)
        for (size_t dimy = 0; dimy < mat.getN(); ++dimy)
          if (mat(dimx, dimy) > ret)
            ret = mat(dimx, dimy);
      return ret;
    } // Calculate the infty norm
    if (target == vector_1) {
      float ret = 0;
      for (size_t dimx = 0; dimx < mat.getM(); ++dimx)
        for (size_t dimy = 0; dimy < mat.getN(); ++dimy)
          ret += mat(dimx, dimy);
      return ret;
    } // Calculate Vector 1-Norm
    if (target == vector_2) {
      float ret = 0;
      for (size_t dimx = 0; dimx < mat.getM(); ++dimx)
        for (size_t dimy = 0; dimy < mat.getN(); ++dimy)
          ret += (mat(dimx, dimy) * mat(dimx, dimy));
      return sqrt(ret);
    } // Calculate Vector 2-Norm
    if (target == vector_p) {
      float ret = 0;
      for (size_t dimx = 0; dimx < mat.getM(); ++dimx)
        for (size_t dimy = 0; dimy < mat.getN(); ++dimy)
          ret += (fastpow(mat(dimx, dimy)), p);
      return std::pow(ret, 1.0 / p);
    } // Calculate Vector p-norm
  } else {
    if (target == matrix_1) {
      float maxsum = 0;
      for (size_t row = 0; row < mat.getM(); ++row)
        maxsum += mat(row, 0);
      for (size_t col = 1; col < mat.getN(); ++col) {
        float sum = 0;
        for (size_t row = 0; row < mat.getM(); ++row)
          sum += mat(row, col);
        if (sum > maxsum)
          maxsum = sum;
      }
      return maxsum; // Calculate Matrix 1-norm
    } else if (target == matrix_2) {
      std::vector<float> matrixU = std::vector<float>();
      std::vector<float> matrixV = std::vector<float>();
      std::vector<float> singularValues = svd(mat, matrixU, matrixV);
      float maxSingularValues = singularValues[0];
      for (auto &j : singularValues)
        if (j > maxSingularValues)
          maxSingularValues = j;
      return maxSingularValues;
    } else if (target == matrix_infty) {
      float maxsum = 0;
      for (size_t col = 0; col < mat.getN(); ++col)
        maxsum += mat(0, col);
      for (size_t row = 1; row < mat.getM(); ++row) {
        float sum = 0;
        for (size_t col = 0; col < mat.getN(); ++col)
          sum += mat(row, col);
        if (sum > maxsum)
          maxsum = sum;
      }
      return maxsum; // Calculate Matrix infty-norm
    } else if (target == matrix_F) {
      float sum = 0;
      for (size_t row = 0; row < mat.getM(); ++row)
        for (size_t col = 0; col < mat.getN(); ++col)
          sum += (mat(row, col) * mat(row, col));
      return std::sqrt(sum); // Calculate Matrix F-Norm
    }
  }
}
template <class T> matrix<T> getTranspose(const matrix<T> &mat) {
  matrix<T> ret(mat.getN(), mat.getM());
  for (size_t row = 0; row < mat.getN(); ++row)
    for (size_t col = 0; col < mat.getM(); ++col)
      ret(row, col) = mat(col, row);
  return ret;
}
template <class T> matrix<float> linear(const matrix<T>& src){
  return src;
}

template <class T> float _tanh(const T& data){
  checkTypeCalculated(T());
  return tanh(data);
}

template <class T> matrix<float> tanh(const matrix<T>& src){
  matrix<float> result(src.getM(), src.getN());
  if (src.getN() != 1 && src.getM() != 1)
    std::cerr << "Warning: [" << __FILE__ << ":" << __LINE__
              << "]: Try to do tanh on a matrix object" << std::endl;
  for (size_t idx = 0; idx < src.getM(); ++idx)
    for (size_t idx2 = 0; idx2 < src.getN(); ++idx2)
      result(idx, idx2) = _tanh(src(idx, idx2));
  return result;
  // Todo: Tobe tested
}

template <class T> matrix<float> tanhGrad(const matrix<T>& src){
  matrix<float> result(src.getM(), src.getN());
  if (src.getN() != 1 && src.getM() != 1)
    std::cerr << "Warning: [" << __FILE__ << ":" << __LINE__
              << "]: Try to do tanh on a matrix object" << std::endl;
  for (size_t idx = 0; idx < src.getM(); ++idx)
    for (size_t idx2 = 0; idx2 < src.getN(); ++idx2)
      result(idx, idx2) = 1 - (_tanh(src(idx, idx2)) * _tanh(src(idx, idx2)));
  return result;
  // Todo: Tobe tested
}


template <class T> float _relu(const T& data){
  checkTypeCalculated(T());
  return data > T(0) ? data : T(0);
}

template <class T> matrix<float> relu(const matrix<T>& src){
  matrix<float> result(src.getM(), src.getN());
  if (src.getN() != 1 && src.getM() != 1)
    std::cerr << "Warning: [" << __FILE__ << ":" << __LINE__
              << "]: Try to do relu on a matrix object" << std::endl;
  for (size_t idx = 0; idx < src.getM(); ++idx)
    for (size_t idx2 = 0; idx2 < src.getN(); ++idx2)
      result(idx, idx2) = _relu(src(idx, idx2));
  return result;
  // Todo: Tobe tested
}
template <class T> float _sigmoid(const T &data) {
  checkTypeCalculated(T());
  return 1.0 / (1.0 + exp(-data));
}

template <class T> matrix<float> sigmoidGrad(const matrix<T> &src) {
  matrix<float> result(src.getM(), src.getN());
  if (src.getN() != 1 && src.getM() != 1)
    std::cerr << "Warning: [" << __FILE__ << ":" << __LINE__
              << "]: Try to do sigmoid on a matrix object" << std::endl;
  for (size_t idx = 0; idx < src.getM(); ++idx)
    for (size_t idx2 = 0; idx2 < src.getN(); ++idx2)
      result(idx, idx2) = (_sigmoid(src(idx, idx2))) * (1 - _sigmoid(src(idx, idx2)));
  return result;
  // Todo: Tobe tested
}

template <class T> matrix<float> sigmoid(const matrix<T> &src) {
  matrix<float> result(src.getM(), src.getN());
  if (src.getN() != 1 && src.getM() != 1)
    std::cerr << "Warning: [" << __FILE__ << ":" << __LINE__
              << "]: Try to do sigmoid on a matrix object" << std::endl;
  for (size_t idx = 0; idx < src.getM(); ++idx)
    for (size_t idx2 = 0; idx2 < src.getN(); ++idx2)
      result(idx, idx2) = _sigmoid(src(idx, idx2));
  return result;
  // Todo: Tobe tested
}
template <class T> matrix<float> resizeVector(const matrix<T> &src) {
  bool flag = false; // false if it is matrix, true if it is vector
  if (src.getN() != 1 && src.getM() != 1)
    flag = false;
  else
    flag = true;
  matrix<float> ret(src);
  float l2norm =
      flag ? norm(src, NORM::vector_2, -1) : norm(src, NORM::matrix_F, -1);
  for (size_t idx1 = 0; idx1 < src.getM(); ++idx1)
    for (size_t idx2 = 0; idx2 < src.getN(); ++idx2)
      ret(idx1, idx2) = 1.0 * src(idx1, idx2) / l2norm;
  return ret;
}

/*
 * QR_compute_householder_factor: mat = I - 2*v*v^T
 */
template <class T>
matrix<float> QR_compute_householder_factor(const matrix<T> &mat,
                                            const matrix<T> &vec) {
  if (vec.getN() != 1 && vec.getM() != 1)
    throw exception("Householder should be vector", std::string(__FILE__),
                    "MatrixError(QR)", __LINE__);
  matrix<float> rawVec = vec.getM() == 1 ? getTranspose(vec) : vec;
  matrix<float> transposedVec = getTranspose(rawVec);
  matrix<float> ret(rawVec.getM(), rawVec.getM());

  for (size_t idx1 = 0; idx1 < rawVec.getM(); ++idx1)
    for (size_t idx2 = 0; idx2 < rawVec.getM(); ++idx2)
      ret(idx1, idx2) = -2 * rawVec(idx1, 0) * rawVec(0, idx2);

  ret += getIdentity<float>(rawVec.getM());
  return ret;
}
template <class T>
matrix<T> getRow(const matrix<T>& src, const size_t& row) {
  if(row >= src.getM())
    throw exception("Row larger than matrix! [" +
                    std::to_string(src.getM()) + "] Request [" +
                    std::to_string(row) + "]",
                    std::string(__FILE__), "MatrixError", __LINE__);
  matrix<T> ret(1, src.getN());
  for(size_t idx = 0; idx < src.getN(); ++idx)
    ret(0, idx) = src(row, idx);
  return ret;
}

template <class T>
matrix<T> getColumn(const matrix<T> &src, const size_t &col) {
  if (col >= src.getN())
    throw exception("Column larger than matrix! [" +
                        std::to_string(src.getN()) + "] Request [" +
                        std::to_string(col) + "]",
                    std::string(__FILE__), "MatrixError", __LINE__);
  matrix<T> ret(src.getM(), 1);
  for (size_t idx = 0; idx < src.getM(); ++idx)
    ret(idx, 0) = src(idx, col);
  return ret;
}

template <class T>
matrix<T> getMinorMatrix(const matrix<T> &src, const size_t &d) {
  matrix<T> ret = src;
  for (size_t idx = 0; idx < d; ++idx)
    for (size_t idx2 = 0; idx2 < d; ++idx2)
      ret(idx, idx2) = (idx == idx2) ? T(1) : T(0);
  return ret;
}

template <class T>
std::pair<matrix<float>, matrix<float>>
QR_HouseHolder(const matrix<T> &src, matrix<float> &R, matrix<float> &Q) {
  size_t m = src.getM();
  size_t n = src.getN();
  std::vector<matrix<float>> qv(m);
  matrix<float> z(src);
  matrix<float> z1(z);
  for (size_t k = 0; k < n && k < m - 1; ++k) {
    matrix<float> e(m, 1), x(m, 1);
    float a;
    z1 = getMinorMatrix(z, k);
    x = getColumn(z1, k);
    a = norm(x, NORM::vector_2);
    if (src(k, k) > 0)
      a = -a;
    for (size_t idx = 0; idx < m; ++idx) {
      e(idx, 0) = (idx == k) ? 1.0 * a : 0.0; // e = a * e
    }
    e = resizeVector(x + e); // e = x + a * e'
    qv[k] = QR_compute_householder_factor(qv[k], e);
    z = qv[k] * z1;
  }
  Q = qv[0];
  for (int i = 1; i < n && i < m - 1; i++) {
    z1 = qv[i] * Q;
    Q = z1;
  }
  R = Q * src;
  Q.transpose();
  return std::make_pair(Q, R);
}

template <class T>
std::pair<matrix<float>, matrix<float>> LU_Decomposition(const matrix<T> &src) {
  if (src.getN() != src.getM())
    throw exception(
        "Not a square matrix, LU Decomposition can only apply to square matrix",
        std::string(__FILE__), "MatrixError(LU)", __LINE__);
  matrix<float> l(src.getM(), src.getM()), u(src.getM(), src.getM());
  for(size_t i = 0; i < src.getM(); ++i)
    for(size_t j = 0; j < src.getM(); ++j){
      if(j < i) l(j, i) = 0;
      else {
        l(j, i) = src(j, i);
        for(size_t k = 0; k < i; ++k){
          l(j, i) = l(j, i) - l(j, k) * u(k, i);
        }
      }
      for (size_t j = 0; j < src.getN(); j++) {
        if (j < i) u(i, j) = 0;
        else if (j == i)
          u(i, j) = 1;
        else {
          u(i, j) = 1. * src(i, j) / l(i, i);
          for (size_t k = 0; k < i; k++) {
            u(i, j) = 1. * u(i, j) - ((1. * l(i, k) * u(k, j)) / l(i, i));
          }
        }
      }
    }
  return std::make_pair(l, u);

}

} // namespace matrix
} // namespace peterzheng

#endif // MLINCPP_MATRIXTOOLS_H
