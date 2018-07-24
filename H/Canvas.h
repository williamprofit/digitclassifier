#ifndef H_CANVAS
#define H_CANVAS

#include <SFML/Graphics.hpp>
#include <Eigen/Dense>

class Canvas
{
public:
	Canvas();
	Canvas(float sizeX, float sizeY, float resolutionX, float resolutionY);
	~Canvas();

	virtual void draw(sf::RenderWindow& window);

	virtual void setPosition(float x, float y);
	virtual void setSize(float x, float y);
	virtual void setResolution(float x, float y);
	virtual void setBrushRadius(float radius);

	virtual void setColorForeground(sf::Color color);
	virtual void setColorBackground(sf::Color color);

	virtual void clear();

	virtual void displayImage(const Eigen::VectorXf& image, int sizeX, int sizeY);

	virtual Eigen::MatrixXf& getPixels();
	virtual Eigen::VectorXf getPixelsAsVec();

protected:
	virtual void checkForClick(sf::Window& window);
	virtual bool isClickInCanvas(const sf::Vector2i& clickPos);
	virtual void drawOnCanvas(float posX, float posY);

	sf::Vector2f m_position;
	sf::Vector2f m_size;
	sf::Vector2f m_resolution;

	float m_brushRadius;

	sf::Color m_colorForeground;
	sf::Color m_colorBackground;

	sf::RectangleShape m_background;
	sf::RectangleShape m_foreground;

	Eigen::MatrixXf m_pixels;
};

#endif