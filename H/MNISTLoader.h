#ifndef H_MNIST_LOADER
#define H_MNIST_LOADER

#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>

class MNISTLoader
{
public:
	MNISTLoader(bool loadTrain = true, bool loadTest = true);
	~MNISTLoader();

	virtual void load(std::string dirPath);

	virtual std::vector<Eigen::VectorXf>& getTrainImages();
	virtual std::vector<Eigen::VectorXf>& getTrainLabels();

	virtual std::vector<Eigen::VectorXf>& getTestImages();
	virtual std::vector<Eigen::VectorXf>& getTestLabels();

protected:
	virtual void loadImages(std::string filePath, std::vector<Eigen::VectorXf>& vec);
	virtual void loadLabels(std::string filePath, std::vector<Eigen::VectorXf>& vec);

	virtual Eigen::VectorXf loadImageFromBytes(const std::vector<unsigned char>& bytes, int32_t nbRows, int32_t nbCols, int32_t offset);
	virtual Eigen::VectorXf loadLabelFromBytes(const std::vector<unsigned char>& bytes, int32_t offset);

	virtual void readAllBytes(std::string filePath, std::vector<unsigned char>& output);
	virtual int32_t cast4BytesToInt32(const std::vector<unsigned char>& bytes, unsigned int start);

	std::vector<Eigen::VectorXf> m_trainImages;
	std::vector<Eigen::VectorXf> m_trainLabels;

	std::vector<Eigen::VectorXf> m_testImages;
	std::vector<Eigen::VectorXf> m_testLabels;

	bool m_loadTrain;
	bool m_loadTest;
};

#endif