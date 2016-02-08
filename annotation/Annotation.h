#ifndef ANNOTATION_H
#define ANNOTATION_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "core/Point.h"
#include "config/pathology_config.h"

class AnnotationGroup;

class EXPORT_PATHOLOGYANNOTATION Annotation {
public:

  enum Type {
    NONE,
    DOT,
    POLYGON,
    SPLINE,
    POINTSET
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

	void setName(const std::string& name);
  std::string getName() const;

  void setGroup(const std::shared_ptr<AnnotationGroup>& group);
  std::shared_ptr<AnnotationGroup> getGroup() const;

  std::string getColor() const;
  void setColor(const std::string& color);

  std::vector<Point> getImageBoundingBox() const;
  std::vector<Point> getLocalBoundingBox();
  Point getCenter();

  void simplify(unsigned int nrPoints);

  float getArea() const;
  unsigned int getNumberOfPoints() const;

private:
  Type _type;
	std::vector<Point> _coordinates;
  std::string _name;
  std::weak_ptr<AnnotationGroup> _group;
  static const char* _typeStrings[5];
  std::string _color;
};
#endif