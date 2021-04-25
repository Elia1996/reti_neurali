cmake_minimum_required(VERSION 3.3)
project(reteNeurale4)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp include.hpp Perceptron.hpp Neural.hpp)
add_executable(reteNeurale4 ${SOURCE_FILES})