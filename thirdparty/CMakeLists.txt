cmake_minimum_required(VERSION 3.1)

# - Eigen3 - #
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/Eigen3)
target_include_directories(DigitClassifier PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Eigen3)
