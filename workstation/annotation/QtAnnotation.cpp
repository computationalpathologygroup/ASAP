#include "QtAnnotation.h"
#include "Annotation.h"

using namespace std;

QtAnnotation::QtAnnotation(Annotation* annotation, float scale) :
QGraphicsItem(),
QObject(),
_annotation(annotation),
_scale(scale)
{
  // We consider the first point to act as the 'center' of the annotation
  if (_annotation) {
    Point center = _annotation->getCoordinate(0);
    this->setPos(center.getX()*_scale, center.getY()*_scale);
  }
  this->setFlag(QGraphicsItem::ItemIsSelectable);
}

void QtAnnotation::addCoordinate(const float& x, const float& y) {
  prepareGeometryChange();
  if (_annotation) {
    _annotation->addCoordinate(x, y);
  }
}

void QtAnnotation::addCoordinate(const Point& xy) {
  prepareGeometryChange();
  if (_annotation) {
    _annotation->addCoordinate(xy);
  }
}

void QtAnnotation::insertCoordinate(const int& index, const float& x, const float& y) {
  prepareGeometryChange();
  if (_annotation) {
    _annotation->insertCoordinate(index, x, y);
  }
}

void QtAnnotation::insertCoordinate(const int& index, const Point& xy){
  prepareGeometryChange();
  if (_annotation) {
    _annotation->insertCoordinate(index, xy);
  }
}

void QtAnnotation::removeCoordinate(const int& index) {
  prepareGeometryChange();
  if (_annotation) {
    _annotation->removeCoordinate(index);
  }
}

void QtAnnotation::setCoordinates(const std::vector<Point>& coordinates) {
  prepareGeometryChange();
  if (_annotation) {
    _annotation->setCoordinates(coordinates);
  }
}

Annotation* QtAnnotation::getAnnotation() const {
  return _annotation;
}

void QtAnnotation::setActiveSeedPoint(const unsigned int seedPointIndex) {
  if (seedPointIndex < this->getAnnotation()->getCoordinates().size()) {
    _activeSeedPoint = seedPointIndex;
  }
}

void QtAnnotation::clearActiveSeedPoint() {
  _activeSeedPoint = -1;
}

int QtAnnotation::getActiveSeedPoint() {
  return _activeSeedPoint;
}

void QtAnnotation::setEditable(const bool editable) {
  _editable = editable;
}

bool QtAnnotation::getEditable() const {
  return _editable;
}