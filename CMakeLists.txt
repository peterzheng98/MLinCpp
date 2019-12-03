cmake_minimum_required(VERSION 3.15)
project(MLinCpp)

set(CMAKE_CXX_STANDARD 14)


add_executable(MLinCpp main.cpp matrix.h misc/exception.h misc/usefulTools.h)
add_executable(MatrixTest matrixTest.cpp)