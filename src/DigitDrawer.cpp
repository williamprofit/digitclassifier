#include <DigitDrawer.h>
#include <ctime>
#include <sstream>

DigitDrawer::DigitDrawer(float sizeX, float sizeY, bool loadMNIST) : m_mnist(false, true)
{
	m_window.create(sf::VideoMode(sizeX, sizeY), "Digit Classifier");
	m_canvas.setResolution(28, 28);
	m_canvas.setSize(sizeX, sizeY);

	m_font.loadFromFile("../res/Consolas.ttf");
	m_label.setFont(m_font);
	m_label.setCharacterSize(12);

	m_nn.load("../res/NNsaves/default");

	m_enterTimer = 0.0f;
	m_mnistIndex = 0;
	m_mnistOn = loadMNIST;
	if (m_mnistOn)
		m_mnist.load("../res/MNIST");

	m_run = true;
}

DigitDrawer::~DigitDrawer()
{ 
}

void DigitDrawer::run()
{
	float lastFrameTime = 0.0f;
	sf::Event ev;

	while (m_run)
	{
		while (m_window.pollEvent(ev))
		{
			if (ev.type == sf::Event::Closed)
				m_run = false;
		}

		if (std::clock() - lastFrameTime >= 1.0f / 60.0f)
		{
			lastFrameTime = std::clock();

			m_window.clear();
			this->draw();
			m_window.display();
		}
	}
}

void DigitDrawer::draw()
{
	m_canvas.draw(m_window);

	this->manageKBInput();
	this->feedToNN(m_canvas.getPixelsAsVec());
}

void DigitDrawer::manageKBInput()
{
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
	{
		m_run = false;
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
	{
		m_canvas.clear();
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Enter) && std::clock() - m_enterTimer >= 100)
	{
		m_enterTimer = std::clock();
		m_canvas.displayImage(m_mnist.getTestImages()[m_mnistIndex++], 28, 28);
	}
}

void DigitDrawer::feedToNN(const Eigen::VectorXf& input)
{
	m_nn.feedforward(input);
	this->drawResults(m_nn.getOutput());
}

void DigitDrawer::drawResults(const Eigen::VectorXf& results)
{
	float max = results.maxCoeff();

	for (int i = 0; i < 10; i++)
	{
		m_label.setPosition(0, i * 15);
		
		std::stringstream ss;
		ss << i << ": " << int(results(i) * 100) << '%';
		m_label.setString(ss.str());

		sf::Color color;
		color.r = 100.0f;
		color.g = 100.0f + (results(i) / max) * 155.0f;
		color.b = 100.0f;
		m_label.setColor(color);

		m_window.draw(m_label);
	}
}