#include <EigenFileIO.h>

using namespace Eigen;

void writeMatrixBinary(std::ofstream& stream, const MatrixXf& matrix)
{
	uint32_t rows = matrix.rows();
	uint32_t cols = matrix.cols();

	stream.write((char*)&rows, sizeof(rows));
	stream.write((char*)&cols, sizeof(cols));

	stream.write((char*)matrix.data(), rows * cols * sizeof(float));
}

void writeVectorBinary(std::ofstream& stream, const VectorXf& vector)
{
	uint32_t size = vector.size();
	stream.write((char*)(&size), sizeof(size));

	stream.write((char*)vector.data(), size * sizeof(float));
}


void readMatrixBinary(std::ifstream& stream, MatrixXf& matrix)
{
	uint32_t rows;
	uint32_t cols;

	stream.read((char*)&rows, sizeof(rows));
	stream.read((char*)&cols, sizeof(cols));

	matrix.resize(rows, cols);

	stream.read((char*)matrix.data(), rows * cols * sizeof(float));
}

void readVectorBinary(std::ifstream& stream, VectorXf& vector)
{
	uint32_t size;
	stream.read((char*)&size, sizeof(size));

	vector.resize(size);

	stream.read((char*)vector.data(), size * sizeof(float));
}