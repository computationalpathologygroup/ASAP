#ifndef WSCOORDINATE_H
#define WSCOORDINATE_H
#include "config/pathology_config.h"

class EXPORT_CORE Point {
public:
  Point::Point(const float& x, const float& y);
  inline float getX() const { return _x; }
  inline float getY() const { return _y; };
  inline void  setX(const float& x) { _x = x; }
  inline void  setY(const float& y) { _y = y; }

private:
	float _x;
	float _y;
};
#endif