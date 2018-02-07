#include "DotQtAnnotation.h"
#include "Annotation.h"
#include <QPainter>
#include <QStyleOptionGraphicsItem>

DotQtAnnotation::DotQtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject *parent, float scale) : 
  QtAnnotation(annotation, parent, scale),
  _rectSize(5.),
  _rectColor(QColor("blue")),
  _rectSelectedColor(QColor("red")),
  _currentLOD(1.)
{
}

void DotQtAnnotation::moveCoordinatesBy(const Point& moveBy) {
  moveCoordinateBy(moveBy);
}

void DotQtAnnotation::moveCoordinateBy(const Point& moveBy) {
  std::vector<Point> coords = this->getAnnotation()->getCoordinates();
  prepareGeometryChange();
  coords[0].setX(coords[0].getX() + moveBy.getX() / _scale);
  coords[0].setY(coords[0].getY() + moveBy.getY() / _scale);
  this->getAnnotation()->setCoordinates(coords);
  this->setPos(QPointF(coords[0].getX()*_scale, coords[0].getY()*_scale));
  onAnnotationChanged();
}

QRectF DotQtAnnotation::boundingRect() const {
  QRectF bRect;
  QPointF tl(-1.5*_rectSize, -1.5*_rectSize);
  QPointF br(1.5*_rectSize, 1.5*_rectSize);
  bRect = QRectF(tl, br);
  return bRect;
}

QPainterPath DotQtAnnotation::shape() const {
  QPainterPath path;
  path.addEllipse(QPointF(), 1.5 * _rectSize / _currentLOD, 1.5 * _rectSize / _currentLOD);
  return path;
}

void DotQtAnnotation::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
  QWidget *widget) {
  _currentLOD = option->levelOfDetailFromTransform(painter->worldTransform());
  painter->setPen(Qt::NoPen);
  if (isSelected()) {
    painter->setBrush(QBrush(_rectSelectedColor));
  }
  else {
    painter->setBrush(QBrush(getDrawingColor()));
  }
  if (isSelected()) {
    painter->drawEllipse(QPointF(), 1.5 * _rectSize / _currentLOD, 1.5 * _rectSize / _currentLOD);
  }
  else {
    painter->drawEllipse(QPointF(), _rectSize / _currentLOD, _rectSize / _currentLOD);
  }  
}