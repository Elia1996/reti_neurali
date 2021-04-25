cmake_minimum_required(VERSION 3.6)
project(ReteNeurale2)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp Neural.h Perceptron.h include.h)
add_executable(ReteNeurale2 ${SOURCE_FILES})