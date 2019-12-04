cmake_minimum_required(VERSION 3.15)
project(MLinCpp)

set(CMAKE_CXX_STANDARD 14)


add_executable(MLinCpp main.cpp misc/exception.h misc/usefulTools.h)
add_executable(MatrixTest unitTest/matrixTest.cpp)
add_executable(RNN neuralNetworks/RNN.cpp)