cmake_minimum_required(VERSION 3.15)
project(MLinCpp)

set(CMAKE_CXX_STANDARD 14)


add_executable(MLinCpp main.cpp misc/exception.h misc/usefulTools.h linearRegression/logisticRegression.hpp)
add_executable(MatrixTest unitTest/matrixTest.cpp)
add_executable(RNN neuralNetworks/RNN.cpp neuralNetworks/Conv2D.cpp neuralNetworks/Conv2D.h)
add_executable(RunLR unitTest/runLR.cpp neuralNetworks/Dense.cpp)