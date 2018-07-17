#ifndef H_EIGEN_FILE_IO
#define H_EIGEN_FILE_IO

#include <Eigen/Dense>
#include <fstream>

void writeMatrixBinary(std::ofstream& stream, const Eigen::MatrixXf& matrix);
void writeVectorBinary(std::ofstream& stream, const Eigen::VectorXf& vector);

void readMatrixBinary(std::ifstream& stream, Eigen::MatrixXf& matrix);
void readVectorBinary(std::ifstream& stream, Eigen::VectorXf& vector);

#endif