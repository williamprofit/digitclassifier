cmake_minimum_required(VERSION 3.1)

project(DigitClassifier C CXX)
set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
        "Choose build type: Debug, Release"
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# --- Project --- #

set (H
        H/NeuralNetwork.h
		H/TrainingInfo.h
        H/Layer.h
        H/ActivationFunc.h
		H/ActivationFuncTable.h
        H/SigmoidFunc.h
        H/ReLUFunc.h
		H/TanhFunc.h
		H/SoftmaxFunc.h
		H/LossFunc.h
		H/LossFuncTable.h
		H/CrossEntropyFunc.h
		H/MSEFunc.h
		H/MNISTLoader.h
		H/EigenFileIO.h
		H/DigitDrawer.h
		H/Canvas.h)

set (src
        src/main.cpp
        src/NeuralNetwork.cpp
        src/Layer.cpp
        src/SigmoidFunc.cpp
        src/ReLUFunc.cpp
		src/TanhFunc.cpp
		src/SoftmaxFunc.cpp
		src/CrossEntropyFunc.cpp
		src/MSEFunc.cpp
		src/MNISTLoader.cpp
		src/EigenFileIO.cpp
		src/DigitDrawer.cpp
		src/Canvas.cpp)

# Linking
include_directories(H/)

add_executable(DigitClassifier ${src} ${H})

add_subdirectory(thirdparty)
target_link_libraries(DigitClassifier eigen sfml-system sfml-graphics)