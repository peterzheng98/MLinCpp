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
namespace peterzheng {
namespace matrix {
template <class T>
std::vector<float> eigen(const matrix<T>& mat, matrix<float>& V, matrix<float>& U){
  // Todo: Eigen Value Decomposition
  return std::vector<float>();
}

template<class T>
std::vector<float> svd(const matrix<T>& mat, matrix<float>& V, matrix<float>& U){
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
      for(auto &j : singularValues)
        if(j > maxSingularValues) maxSingularValues = j;
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
    } else if(target == matrix_F){
      float sum = 0;
      for (size_t row = 0; row < mat.getM(); ++row)
        for (size_t col = 0; col < mat.getN(); ++col)
          sum += (mat(row, col) * mat(row, col));
      return std::sqrt(sum); // Calculate Matrix F-Norm
    }
  }
}
template <class T>
matrix<T> getTranspose(const matrix<T>& mat){
  matrix<T> ret(mat.getN(), mat.getM());
  for(size_t row = 0; row < mat.getN(); ++row)
    for(size_t col = 0; col < mat.getM(); ++col)
      ret(row, col) = mat(col, row);
  return ret;
}

} // namespace matrix
} // namespace peterzheng

#endif // MLINCPP_MATRIXTOOLS_H
