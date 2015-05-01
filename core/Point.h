#ifndef WSCOORDINATE_H
#define WSCOORDINATE_H
#include "config/pathology_config.h"

class EXPORT_CORE Point {
public:
  Point::Point(const float& x, const float& y);
  float getX() const;
  float getY() const;
  void  setX(const float& x);
  void  setY(const float& y);

private:
	float _x;
	float _y;
};
#endif