cmake_minimum_required(VERSION 3.1)

# - Eigen3 - #
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/Eigen3)
target_include_directories(DigitClassifier PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/Eigen3)

# - SFML - #
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/SFML)
target_include_directories(DigitClassifier PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/SFML/include)

set(BUILD_TESTING OFF CACHE BOOL OFF FORCE)
set(SFML_BUILD_AUDIO OFF CACHE BOOL OFF FORCE)
set(SFML_BUILD_NETWORK OFF CACHE BOOL OFF FORCE)
