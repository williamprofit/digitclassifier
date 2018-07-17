#include <MNISTLoader.h>
#include <iostream>

using namespace Eigen;

MNISTLoader::MNISTLoader()
{
}

MNISTLoader::~MNISTLoader()
{
}

void MNISTLoader::load(std::string dirPath)
{
	std::cout << "Loading MNIST dataset ...\n";

	this->loadImages(dirPath + "/train-images-idx3-ubyte", m_trainImages);
	this->loadLabels(dirPath + "/train-labels-idx1-ubyte", m_trainLabels);

	this->loadImages(dirPath + "/t10k-images-idx3-ubyte", m_testImages);
	this->loadLabels(dirPath + "/t10k-labels-idx1-ubyte", m_testLabels);

	std::cout << "Done.\n";
}

void MNISTLoader::loadImages(std::string filePath, std::vector<Eigen::VectorXf>& vec)
{
	std::vector<unsigned char> bytes;
	this->readAllBytes(filePath, bytes);

	/* Magic number takes up first 4 bytes and is equal to 2051 */
	int32_t magicNumber = this->cast4BytesToInt32(bytes, 0);

	if (magicNumber != 2051)
	{
		std::cout << "MNIST file " << filePath << " is corrupt, impossible to load MNIST dataset.\n";
		return;
	}

	int32_t nbImages = this->cast4BytesToInt32(bytes, 4);
	int32_t nbRows	 = this->cast4BytesToInt32(bytes, 8);
	int32_t nbCols   = this->cast4BytesToInt32(bytes, 12);

	int32_t offset = 16;
	for (int32_t i = 0; i < nbImages; i++)
	{
		VectorXf image = this->loadImageFromBytes(bytes, nbRows, nbCols, offset);
		image /= 255.0f;
		vec.push_back(image);

		offset += nbRows * nbCols;
	}
}

void MNISTLoader::loadLabels(std::string filePath, std::vector<Eigen::VectorXf>& vec)
{
	std::vector<unsigned char> bytes;
	this->readAllBytes(filePath, bytes);

	/* Magic number takes up first 4 bytes and is equal to 2049 */
	int32_t magicNumber = this->cast4BytesToInt32(bytes, 0);

	if (magicNumber != 2049)
	{
		std::cout << "MNIST file " << filePath << " is corrupt, impossible to load MNIST dataset.\n";
		return;
	}

	int32_t nbLabels = this->cast4BytesToInt32(bytes, 4);

	int32_t offset = 8;
	for (int32_t i = 0; i < nbLabels; i++)
	{
		VectorXf label = this->loadLabelFromBytes(bytes, offset);
		vec.push_back(label);

		offset++;
	}
}

VectorXf MNISTLoader::loadImageFromBytes(const std::vector<unsigned char>& bytes, int32_t nbRows, int32_t nbCols, int32_t offset)
{
	VectorXf image(nbRows * nbCols);

	for (int i = 0; i < nbRows * nbCols; i++)
		image[i] = (float)bytes[offset + i];

	return image;
}

VectorXf MNISTLoader::loadLabelFromBytes(const std::vector<unsigned char>& bytes, int32_t offset)
{
	VectorXf label = VectorXf::Zero(10);
	label[bytes[offset]] = 1.0f;

	return label;
}

void MNISTLoader::readAllBytes(std::string filePath, std::vector<unsigned char>& output)
{
	std::ifstream file(filePath, std::ios::binary | std::ios::ate);

	if (!file.is_open())
	{
		std::cout << "Failed to load MNIST dataset, could not open file " << filePath << "\n";
		return;
	}

	std::ifstream::pos_type length = file.tellg();
	output.resize(length);
	file.seekg(0, std::ios::beg);

	file.read((char*)&output[0], length);
	file.close();
}

int32_t MNISTLoader::cast4BytesToInt32(const std::vector<unsigned char>& bytes, unsigned int start)
{
	int32_t intOut = 0;
	for (unsigned int i = 0; i < 4; i++)
		intOut += bytes[start + i] << (24 - i * 8);

	return intOut;
}

std::vector<Eigen::VectorXf>& MNISTLoader::getTrainImages()
{
	return m_trainImages;
}

std::vector<Eigen::VectorXf>& MNISTLoader::getTrainLabels()
{
	return m_trainLabels;
}

std::vector<Eigen::VectorXf>& MNISTLoader::getTestImages()
{
	return m_testImages;
}

std::vector<Eigen::VectorXf>& MNISTLoader::getTestLabels()
{
	return m_testLabels;
}

