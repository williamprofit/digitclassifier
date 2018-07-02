# Install script for directory: /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/Cholesky"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/CholmodSupport"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/Core"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/Dense"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/Eigen"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/Eigenvalues"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/Geometry"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/Householder"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/IterativeLinearSolvers"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/Jacobi"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/LU"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/MetisSupport"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/OrderingMethods"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/PaStiXSupport"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/PardisoSupport"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/QR"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/QtAlignedMalloc"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/SPQRSupport"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/SVD"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/Sparse"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/SparseCholesky"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/SparseCore"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/SparseLU"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/SparseQR"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/StdDeque"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/StdList"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/StdVector"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/SuperLUSupport"
    "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/UmfPackSupport"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE DIRECTORY FILES "/home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

