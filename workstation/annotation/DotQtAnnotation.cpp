#include "DotQtAnnotation.h"
#include "Annotation.h"
#include <QPainter>
#include <QStyleOptionGraphicsItem>

DotQtAnnotation::DotQtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject *parent, float scale) : 
  QtAnnotation(annotation, parent, scale),
  _rectSize(3.),
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
  onCoordinatesChanged();
}

QRectF DotQtAnnotation::boundingRect() const {
  QRectF bRect;
  if (_annotation) {
    QPointF tl(-4.5*_rectSize / _currentLOD, -4.5*_rectSize / _currentLOD);
    QPointF br(4.5*_rectSize / _currentLOD, 4.5*_rectSize / _currentLOD);
    bRect = QRectF(tl, br);
  }
  return bRect;
}

void DotQtAnnotation::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
  QWidget *widget) {
  if (_annotation) {
    _currentLOD = option->levelOfDetailFromTransform(painter->worldTransform());
    std::vector<Point> coords = _annotation->getCoordinates();
    for (std::vector<Point>::const_iterator it = coords.begin(); it != coords.end(); ++it) {
      if (isSelected()) {
        painter->setPen(QPen(QBrush(_rectSelectedColor), 4.5 * _rectSize / _currentLOD, Qt::PenStyle::SolidLine, Qt::PenCapStyle::SquareCap));
      }
      else {
        painter->setPen(QPen(QBrush(getDrawingColor()), 3 * _rectSize / _currentLOD, Qt::PenStyle::SolidLine, Qt::PenCapStyle::SquareCap));
      }
      painter->drawPoint(QPointF());
    }
  }
}