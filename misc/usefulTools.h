//
// Created by 郑文鑫 on 2019/12/2.
//

#ifndef MLINCPP_USEFULTOOLS_H
#define MLINCPP_USEFULTOOLS_H
#include <type_traits>
template <class T> void checkTypeCalculated(const T &t) {
  static_assert(std::is_arithmetic<T>::value, "Not supported Calculation");
}

template <class T> T fastpow(T x, int n){
  checkTypeCalculated(T());
  T res = T(1);
  while (n) {
    if (n & 1)
      res *= x;
    x *= x;
    n >>= 1;
  }
  return res;
}

#endif // MLINCPP_USEFULTOOLS_H
