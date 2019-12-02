//
// Created by 郑文鑫 on 2019/12/2.
//

#ifndef MLINCPP_USEFULTOOLS_H
#define MLINCPP_USEFULTOOLS_H
#include <type_traits>
template <class T> void checkTypeCalculated(const T &t) {
  static_assert(std::is_arithmetic<T>::value, "Not supported Calculation");
}

#endif // MLINCPP_USEFULTOOLS_H
