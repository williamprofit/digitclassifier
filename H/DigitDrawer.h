#ifndef H_DIGIT_DRAWER
#define H_DIGIT_DRAWER

#include <NeuralNetwork.h>
#include <MNISTLoader.h>
#include <Canvas.h>

class DigitDrawer
{
public:
	DigitDrawer(float sizeX, float sizeY, bool loadMNIST = false);
	~DigitDrawer();

	virtual void run();

protected:
	virtual void draw();
	virtual void manageKBInput();
	virtual void feedToNN(const Eigen::VectorXf& input);
	virtual void drawResults(const Eigen::VectorXf& results);

	sf::RenderWindow m_window;
	Canvas m_canvas;
	sf::Font m_font;
	sf::Text m_label;
	bool m_run;

	NeuralNetwork m_nn;

	bool m_mnistOn;
	MNISTLoader m_mnist;
	float m_enterTimer;
	int m_mnistIndex;
};

#endif