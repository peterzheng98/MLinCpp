//
// Created by 郑文鑫 on 2019/12/3.
//
#include "../matrix/matrix.h"
#include "../matrix/matrixTools.h"
#include "../linearRegression/linearRegression.hpp"
#include <fstream>
#include <iostream>
// Stage 1
#include<cmath>
#include <iostream>
#include <vector>
#include <cstring>
using namespace std;

class Matrix {
public:
  vector<vector<double>> A;
  int row;
  int col;
public:
  Matrix() {
    row = 0; col = 0;
  }
  Matrix(vector<vector<double>> a) {
    A.resize(a.size(), vector<double>(a[0].size()));
    for (int i = 0; i < a.size(); i++) {
      A[i].assign(a[i].begin(), a[i].end());
    }
    row= A.size();
    col = A[0].size();
  }
  Matrix Divide(double data) {
    vector<vector<double>> ans(row,vector<double>(col));
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        ans[i][j] = A[i][j] / data;
      }
    }
    return Matrix(ans);
  }
  Matrix Multi(double data) {
    vector<vector<double>> ans(row, vector<double>(col));
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        ans[i][j] = A[i][j] *data;
      }
    }
    return Matrix(ans);
  }
  Matrix Transposition() {//◊™÷√
    vector<vector<double>>ans(col, vector<double>(row));
    for (int i = 0; i < col; i++) {
      for (int j = 0; j < row; j++) {
        ans[i][j] = A[j][i];
      }
    }
    return Matrix(ans);
  }
  Matrix& operator=(const Matrix& r) {
    if (this == &r)return *this;
    A.resize(r.row, vector<double>(r.col));
    row = r.row; col = r.col;
    for (int i = 0; i < r.A.size(); i++) {
      A[i].assign(r.A[i].begin(), r.A[i].end());
    }
    return *this;
  }
  Matrix operator+(const Matrix& r) const {
    vector<vector<double>>C;
    if (row == r.row && col ==r. col) {
      C.resize(row, vector<double>(col));
      for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
          C[i][j] = A[i][j] + r.A[i][j];
        }
      }
      Matrix ans(C);
      return ans;
    }
    else {
      throw"invalid Matrix";
    }
  }
  Matrix operator-(const Matrix& r) const {
    vector<vector<double>>C;
    if (row == r.row && col == r.col) {
      C.resize(row, vector<double>(col));
      for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
          C[i][j] = A[i][j] - r.A[i][j];
        }
      }
      Matrix ans(C);
      return ans;
    }
    else {
      throw"invalid Matrix";
    }
  }
  Matrix operator*(const Matrix& r)const {
    vector<vector<double>> C;//ans
    if (col == r.row) {
      C.resize(row, vector<double>(r.col));
      for (int i = 0; i < C.size(); i++) {
        fill(C[i].begin(), C[i].end(), 0);
      }
      for (int i = 0; i < row; i++) {
        for (int j = 0; j < r.col; j++) {
          for (int k = 0; k < col; k++) {
            C[i][j] += A[i][k] * r.A[k][j];
          }
        }
      }
      Matrix ans(C);
      return ans;
    }
    else {
      throw "invalid Matrix";
    }
  }
  void print() {
    for (int i = 0; i <row; i++) {
      for (int j = 0; j < col; j++) {
        cout << A[i][j] << ' ';
      }
      cout << '\n';
    }
  }
  void dumpToFile(const std::string &name,
                  const std::ios_base::openmode &mode) {
    std::fstream f1(name, mode);
    for (size_t i = 0; i < row; ++i) {
      for (size_t j = 0; j < col; ++j)
        f1 << A[i][j] << " ";
      f1 << std::endl;
    }
    f1 << std::endl;
    f1.close();
  }
};



void test1() {
  std::fstream f1("configure.in"), f2("matrix.in");
  int m, n, op1, op2;
  std::cin >> m >> n >> op1 >> op2;
  f1.close();
  vector<vector<double>> r1, r2;
  r1.resize(m);
  r2.resize(m);
  for(auto &j : r1) j.resize(n);
  for(auto &j : r2) j.resize(n);
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {
      std::cin >> r1[i][j];
    }
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {
      std::cin >> r2[i][j];
    }
  f2.close();
  Matrix m1(r1), m2(r2), p, mi, t1, t2, t3;
  p = m1 + m2;
  p.print();
  std::cout << std::endl;
  mi = m1 - m2;
  mi.print();
  std::cout << std::endl;
  t1 = m1.Multi(r2[0][0]);
  t1.print();
  std::cout << std::endl;
  t2 = m2.Multi(r1[0][0]);
  t2.print();
  std::cout << std::endl;
  t3 = m1 * m2;
  t3.print();

}

int main() { test1(); }