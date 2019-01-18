#include "PointSetQtAnnotation.h"
#include "annotation/Annotation.h"
#include <QPainter>
#include <QPainterPath>
#include <QPainterPathStroker>
#include <QStyleOptionGraphicsItem>
#include <iostream>
#include <cmath>

PointSetQtAnnotation::PointSetQtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject* parent, float scale) : 
  QtAnnotation(annotation, parent, scale),
  _rectColor(QColor("blue")),
  _rectSelectedColor(QColor("red")),
  _currentLoD(1.0),
  _rectSize(3.)
{
  onAnnotationChanged();
}

QRectF PointSetQtAnnotation::boundingRect() const {  
  return _bRect;
}

void PointSetQtAnnotation::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
  QWidget *widget) {
  if (_annotation) {
    _currentLoD = option->levelOfDetailFromTransform(painter->worldTransform());
    //painter->drawRect(_bRect);
    std::vector<Point> coords = _annotation->getCoordinates();
    if (isSelected()) {
      painter->setPen(QPen(QBrush(_rectColor.lighter(150)), 4.5 * _rectSize / _currentLoD, Qt::PenStyle::SolidLine, Qt::PenCapStyle::SquareCap));
    }
    else {
      painter->setPen(QPen(QBrush(_rectColor), 3 * _rectSize / _currentLoD, Qt::PenStyle::SolidLine, Qt::PenCapStyle::SquareCap));
    }
    for (unsigned int i = 0; i < coords.size(); ++i) {
      if (i == _activeSeedPoint) {
        painter->save();
        painter->setPen(QPen(QBrush(_rectSelectedColor), 4.5 * _rectSize / _currentLoD, Qt::PenStyle::SolidLine, Qt::PenCapStyle::SquareCap));
        painter->drawPoint(this->mapFromScene(coords[i].getX()*_scale, coords[i].getY()*_scale));
        painter->restore();
      }
      else {
        QPointF loc = this->mapFromScene(coords[i].getX()*_scale, coords[i].getY()*_scale);
        painter->drawPoint(loc);
      }
    }
  }  
}

bool PointSetQtAnnotation::collidesWithPath(const QPainterPath & path, Qt::ItemSelectionMode mode) const {
  return contains(path.currentPosition());
}

bool PointSetQtAnnotation::contains(const QPointF & point) const {
  if (shape().controlPointRect().contains(point)) {
    QPointF imgPoint = this->mapToScene(point) / _scale;
    double curSelectionSensitivity = (selectionSensitivity * 1.5*_rectSize / _currentLoD);
    double curSelectionSensitivitySquared = curSelectionSensitivity * curSelectionSensitivity;
    double imgX = imgPoint.x();
    double imgY = imgPoint.y();
    std::vector<Point> coords = _annotation->getCoordinates();
    double minDist = std::numeric_limits<double>::max();
    _lastClickedFirstCoordinateIndex = -1;

    // Quickly check if a seed point was hit
    for (unsigned int i = 0; i < coords.size(); ++i) {
      Point pt1 = coords[i];
      double coord1X = pt1.getX(); double coord1Y = pt1.getY();
      double distSquared = pow(imgX - coord1X, 2) + pow(imgY - coord1Y, 2);
      if (distSquared < curSelectionSensitivitySquared && distSquared < minDist) {
        _lastClickedFirstCoordinateIndex = i;
        minDist = distSquared;
      }
    }
    if (_lastClickedFirstCoordinateIndex >= 0) {
      return true;
    }
  }
  return false;
}

void PointSetQtAnnotation::onAnnotationChanged() {
  if (_annotation) {
    std::vector<Point> localBBox = _annotation->getImageBoundingBox();
    QPointF tl = this->mapFromScene(QPointF(localBBox[0].getX() * _scale, localBBox[0].getY() * _scale));
    QPointF br = this->mapFromScene(QPointF(localBBox[1].getX() * _scale, localBBox[1].getY() * _scale));
    tl += QPointF(-4.5*_rectSize / _currentLoD, -4.5*_rectSize / _currentLoD);
    br += QPointF(4.5*_rectSize / _currentLoD, 4.5*_rectSize / _currentLoD);
    _bRect = QRectF(tl, br);
  }
}