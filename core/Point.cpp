#include <Point.h>

Point::Point(const float& x, const float& y)
{
	_x = x;
	_y = y;
};

float Point::getX() const
{
	return _x;
};

float Point::getY() const
{
	return _y;
};

void Point::setX(const float& x) {
  _x = x;
}

void Point::setY(const float& y) {
  _y = y;
}