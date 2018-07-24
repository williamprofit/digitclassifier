#include <Canvas.h>

Canvas::Canvas()
{
	m_background.setFillColor(sf::Color::Black);
	m_foreground.setFillColor(sf::Color::White);

	m_brushRadius = 1.0f;
}

Canvas::Canvas(float sizeX, float sizeY, float resolutionX, float resolutionY) : Canvas()
{
	this->setSize(sizeX, sizeY);
	this->setResolution(resolutionX, resolutionY);
}

Canvas::~Canvas()
{
}

void Canvas::draw(sf::RenderWindow& window)
{
	this->checkForClick(window);

	m_background.setPosition(m_position);
	window.draw(m_background);

	for (unsigned int x = 0; x < m_pixels.rows(); x++)
	{
		for (unsigned int y = 0; y < m_pixels.cols(); y++)
		{
			if (m_pixels(x, y) == 0.0f)
				continue;

			float posX = m_position.x + x * m_foreground.getGlobalBounds().width;
			float posY = m_position.y + y * m_foreground.getGlobalBounds().height;
			m_foreground.setPosition(posX, posY);

			sf::Color color;
			color.r = m_pixels(x, y) * 255.0f;
			color.g = m_pixels(x, y) * 255.0f;
			color.b = m_pixels(x, y) * 255.0f;
			m_foreground.setFillColor(color);

			window.draw(m_foreground);
		}
	}
}

void Canvas::checkForClick(sf::Window& window)
{
	if (!sf::Mouse::isButtonPressed(sf::Mouse::Left))
		return;

	sf::Vector2i clickPos = sf::Mouse::getPosition(window);

	if (!this->isClickInCanvas(clickPos))
		return;

	/* make clickPos relative to canvas */
	clickPos.x -= m_position.x;
	clickPos.y -= m_position.y;

	sf::Vector2f clickedPixel;
	clickedPixel.x = std::floor(clickPos.x / (m_size.x / m_resolution.x));
	clickedPixel.y = std::floor(clickPos.y / (m_size.y / m_resolution.y));

	this->drawOnCanvas(clickedPixel.x, clickedPixel.y);
}

void Canvas::drawOnCanvas(float posX, float posY)
{
	int xMin = std::max(0.0f, posX - m_brushRadius);
	int xMax = std::min(m_resolution.x - 1, posX + m_brushRadius);
	int yMin = std::max(0.0f, posY - m_brushRadius);
	int yMax = std::min(m_resolution.y - 1, posY + m_brushRadius);

	for (int x = xMin; x <= xMax; x++)
	{
		for (int y = yMin; y <= yMax; y++)
		{
			float distance = std::sqrt(std::pow(x - posX, 2) + std::pow(y - posY, 2));
			float ratio = std::max(0.0f, 1.0f / distance);

			m_pixels(x, y) = std::min(1.0f, m_pixels(x, y) + ratio);
		}
	}
}

bool Canvas::isClickInCanvas(const sf::Vector2i& clickPos)
{
	if (clickPos.x <= m_position.x ||
		clickPos.x >= m_position.x + m_size.x ||
		clickPos.y <= m_position.y ||
		clickPos.y >= m_position.y + m_size.y)
	{
		return false;
	}
	else
		return true;
}

void Canvas::clear()
{
	m_pixels.setZero();
}

void Canvas::setPosition(float x, float y)
{
	m_position.x = x;
	m_position.y = y;
}

void Canvas::setSize(float x, float y)
{
	m_size.x = x;
	m_size.y = y;

	m_background.setSize(m_size);

	/* refresh the resolution */
	this->setResolution(m_resolution.x, m_resolution.y);
}

void Canvas::setResolution(float x, float y)
{
	m_resolution.x = x;
	m_resolution.y = y;

	m_pixels.resize(x, y);
	m_pixels.setZero();

	sf::Vector2f foregroundSize;
	foregroundSize.x = m_size.x / m_resolution.x;
	foregroundSize.y = m_size.y / m_resolution.y;
	m_foreground.setSize(foregroundSize);
}

void Canvas::setBrushRadius(float brushRadius)
{
	m_brushRadius = brushRadius;
}

void Canvas::setColorForeground(sf::Color color)
{
	m_foreground.setFillColor(color);
}

void Canvas::setColorBackground(sf::Color color)
{
	m_background.setFillColor(color);
}

void Canvas::displayImage(const Eigen::VectorXf& image, int sizeX, int sizeY)
{
	m_pixels = Eigen::Map<Eigen::MatrixXf>((float*)image.data(), sizeX, sizeY);
}

Eigen::MatrixXf& Canvas::getPixels()
{
	return m_pixels;
}

Eigen::VectorXf Canvas::getPixelsAsVec()
{
	Eigen::VectorXf pixelsVec(Eigen::Map<Eigen::VectorXf>(m_pixels.data(), m_pixels.rows() * m_pixels.cols()));

	return pixelsVec;
}