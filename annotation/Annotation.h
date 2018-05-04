#ifndef ANNOTATION_H
#define ANNOTATION_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "AnnotationBase.h"
#include "core/Point.h"
#include "annotation_export.h"

class AnnotationGroup;

class ANNOTATION_EXPORT Annotation : public AnnotationBase {
public:

  enum Type {
    NONE,
    DOT,
    POLYGON,
    SPLINE,
    POINTSET,
    MEASUREMENT,
    RECTANGLE
  };

  Annotation();

	void addCoordinate(const float& x, const float& y);
  void addCoordinate(const Point& xy);
  void insertCoordinate(const int& index, const float& x, const float& y);
  void insertCoordinate(const int& index, const Point& xy);
  void removeCoordinate(const int& index);
  void setCoordinates(const std::vector<Point>& coordinates);
  Point getCoordinate(const int& index) const;
  std::vector<Point> getCoordinates() const;
  void clearCoordinates();

	void setType(const Annotation::Type& type);
  Annotation::Type getType() const;

  std::string getTypeAsString() const;
  void setTypeFromString(const std::string& type);

  std::vector<Point> getImageBoundingBox() const;
  std::vector<Point> getLocalBoundingBox();
  Point getCenter();

  void simplify(unsigned int nrPoints=0, float epsilon=1.0);

  float getArea() const;
  unsigned int getNumberOfPoints() const;

private:
  Type _type;
	std::vector<Point> _coordinates;
  static const char* _typeStrings[7];
};
#endif