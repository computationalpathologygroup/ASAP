#ifndef ANNOTATION_H
#define ANNOTATION_H

#include <string>
#include <vector>
#include <map>
#include "core/Point.h"
#include "config/pathology_config.h"

class AnnotationGroup;

class EXPORT_PATHOLOGYANNOTATION Annotation {
public:

  enum Type {
    NONE,
    DOT,
    POLYGON,
    SPLINE
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

  void setGroup(AnnotationGroup* group);
  AnnotationGroup* getGroup() const;

  std::string getColor() const;
  void setColor(const std::string& color);

  std::vector<Point> getImageBoundingBox() const;
  std::vector<Point> getLocalBoundingBox();
  Point getCenter();

  void simplify(unsigned int nrPoints);

private:
  Type _type;
	std::vector<Point> _coordinates;
  std::string _name;
  AnnotationGroup* _group;
  static const char* _typeStrings[5];
  std::string _color;
};
#endif