//
// Created by 郑文鑫 on 2019/12/3.
//
#include "../matrix/matrix.h"
#include "../matrix/matrixTools.h"
#include <fstream>
#include <iostream>
// Stage 1
void test1() {
  std::fstream f1("configure.in"), f2("matrix.in");
  int m, n, op1, op2;
  f1 >> m >> n >> op1 >> op2;
  f1.close();
  peterzheng::matrix<int> r1(m, n), r2(m, n);
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {
      f2 >> r1(i, j);
    }
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {
      f2 >> r2(i, j);
    }
  f2.close();
  peterzheng::matrix<int> p, mi, t1, t2, t3;
  p = r1 + r2;
  mi = r1 - r2;
  t1 = r1 * r2(0, 0);
  t2 = r2 * r1(0, 0);
  t3 = r1 * r2;
  p.dumpToFile("result.out", std::ios::out);
  mi.dumpToFile("result.out", std::ios::app);
  t1.dumpToFile("result.out", std::ios::app);
  t2.dumpToFile("result.out", std::ios::app);
  t3.dumpToFile("result.out", std::ios::app);
}

int main() { test1(); }