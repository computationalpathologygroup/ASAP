#include "PolyQtAnnotation.h"
#include "Annotation.h"
#include <QPainter>
#include <QPainterPath>
#include <QPainterPathStroker>
#include <QStyleOptionGraphicsItem>
#include <iostream>

PolyQtAnnotation::PolyQtAnnotation(Annotation* annotation, float scale) : 
  QtAnnotation(annotation, scale),
  _rectSize(10),
  _lineThickness(3),
  _lineAnnotationSelectedThickness(4.5),
  _rectColor(QColor("blue")),
  _rectSelectedColor(QColor("red")),
  _closed(false),
  _type("spline"),
  _currentLoD(1.0)
{

}

QRectF PolyQtAnnotation::boundingRect() const {
  QRectF bRect;
  if (_annotation) {
      QRectF cpRect = getCurrentPath(_annotation->getCoordinates()).controlPointRect();
      QPointF tl = cpRect.topLeft() - QPointF(_rectSize, _rectSize);
      QPointF br = cpRect.bottomRight() + QPointF(_rectSize, _rectSize);
      bRect = bRect.united(QRectF(tl, br));
  }
  return bRect;
}

std::vector<QPointF> PolyQtAnnotation::catmullRomToBezier(const QPointF& p0, const QPointF& p1, const QPointF& p2, const QPointF& p3) const
{
  std::vector<QPointF> bezierPoints;
  bezierPoints.push_back(p1);
  
  float d1 = sqrt(pow(p1.x() - p0.x(), 2) + pow(p1.y() - p0.y(), 2));
  float d2 = sqrt(pow(p2.x() - p1.x(), 2) + pow(p2.y() - p1.y(), 2));
  float d3 = sqrt(pow(p3.x() - p2.x(), 2) + pow(p3.y() - p2.y(), 2));
  float rd1 = sqrt(d1);
  float rd2 = sqrt(d2);
  float rd3 = sqrt(d3);
  float B1X = (d1*p2.x() - d2*p0.x() + (2 * d1 + 3 * rd1*rd2 + d2)*p1.x()) / (3*rd1*(rd1+rd2));
  float B1Y = (d1*p2.y() - d2*p0.y() + (2 * d1 + 3 * rd1*rd2 + d2)*p1.y()) / (3 * rd1*(rd1 + rd2));
  float B2X = (d3*p1.x() - d2*p3.x() + (2 * d3 + 3 * rd3*rd2 + d2)*p2.x()) / (3 * rd3*(rd3 + rd2));
  float B2Y = (d3*p1.y() - d2*p3.y() + (2 * d3 + 3 * rd3*rd2 + d2)*p2.y()) / (3 * rd3*(rd3 + rd2));
  
  bezierPoints.push_back(QPointF(B1X, B1Y));
  bezierPoints.push_back(QPointF(B2X, B2Y));
  bezierPoints.push_back(p2);
  return bezierPoints;
}

void PolyQtAnnotation::setInterpolationType(const std::string& interpolationType) {
  if (interpolationType == "spline") {
    _type = interpolationType;
  }
  else {
    _type = "linear";
  }
}

std::string PolyQtAnnotation::getInterpolationType() {
  return _type;
}

QPainterPath PolyQtAnnotation::getCurrentPath(const std::vector<Point>& coords) const {
  QPainterPath pth;
  pth.moveTo(0, 0);
  for (unsigned int i = 0; i < coords.size() - 1; ++i) {
    if (_type != "spline") {
      pth.lineTo(this->mapFromScene(coords[i + 1].getX()*_scale, coords[i + 1].getY()*_scale));
    }
    else {
      QPointF p1 = this->mapFromScene(coords[i].getX()*_scale, coords[i].getY()*_scale);
      QPointF p2 = this->mapFromScene(coords[i + 1].getX()*_scale, coords[i + 1].getY()*_scale);
      QPointF p0 = p1 - (p2 - p1);
      if (i > 0) {
        p0 = this->mapFromScene(coords[i - 1].getX()*_scale, coords[i - 1].getY()*_scale);
      }
      else if (i == 0 && _closed && coords.size() > 2) {
        p0 = this->mapFromScene(coords[coords.size() - 1].getX()*_scale, coords[coords.size() - 1].getY()*_scale);
      }
      QPointF p3 = p2 + (p2 - p1);
      if (i < coords.size() - 2) {
        p3 = this->mapFromScene(coords[i + 2].getX()*_scale, coords[i + 2].getY()*_scale);
      }
      std::vector<QPointF> bezierPoints = catmullRomToBezier(p0, p1, p2, p3);
      pth.cubicTo(bezierPoints[1], bezierPoints[2], bezierPoints[3]);
    }
  }
  if (_closed) {
    if (_type != "spline") {
      pth.lineTo(0, 0);
    }
    else {
      QPointF p0 = this->mapFromScene(coords[coords.size() - 2].getX()*_scale, coords[coords.size() - 2].getY()*_scale);
      QPointF p1 = this->mapFromScene(coords[coords.size() - 1].getX()*_scale, coords[coords.size() - 1].getY()*_scale);
      QPointF p2 = this->mapFromScene(coords[0].getX()*_scale, coords[0].getY()*_scale);
      QPointF p3 = this->mapFromScene(coords[1].getX()*_scale, coords[1].getY()*_scale);
      std::vector<QPointF> bezierPoints = catmullRomToBezier(p0, p1, p2, p3);
      pth.cubicTo(bezierPoints[1], bezierPoints[2], bezierPoints[3]);
    }
  }
  return pth;
}

void PolyQtAnnotation::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
  QWidget *widget) {
  if (_annotation) {
    QColor lineColor = this->getDrawingColor();
    _currentLoD = option->levelOfDetailFromTransform(painter->worldTransform());
    std::vector<Point> coords = _annotation->getCoordinates();
    if (coords.size() > 1) {
      _currentPath = getCurrentPath(coords);
      painter->setRenderHints(QPainter::Antialiasing);
      painter->setRenderHints(QPainter::HighQualityAntialiasing);
      if (isSelected()) {
        painter->strokePath(_currentPath, QPen(QBrush(lineColor.lighter(150)), _lineAnnotationSelectedThickness / _currentLoD));
      }
      else {
        painter->strokePath(_currentPath, QPen(QBrush(lineColor), _lineThickness / _currentLoD));
      }
    }
    if (isSelected()) {
      painter->setPen(QPen(QBrush(_rectColor.lighter(150)), _lineThickness / _currentLoD));
    }
    else {
      painter->setPen(QPen(QBrush(_rectColor), _lineThickness / _currentLoD));
    }
    for (unsigned int i = 0; i < coords.size(); ++i) {
      if (i == _activeSeedPoint) {
        painter->save();
        painter->setPen(QPen(QBrush(_rectSelectedColor), _lineThickness / _currentLoD));
        painter->drawRect(QRectF(this->mapFromScene(coords[i].getX()*_scale - (_rectSize / _currentLoD) / 2., coords[i].getY()*_scale - (_rectSize / _currentLoD) / 2.), QSizeF(_rectSize / _currentLoD, _rectSize / _currentLoD)));
        painter->restore();
      }
      else {
        painter->drawRect(QRectF(this->mapFromScene(coords[i].getX()*_scale - (_rectSize / _currentLoD) / 2., coords[i].getY()*_scale - (_rectSize / _currentLoD) / 2.), QSizeF(_rectSize / _currentLoD, _rectSize / _currentLoD)));
      }
    }
  }
}

std::pair<int, int> PolyQtAnnotation::seedPointsContainingPathPoint(const QPointF& point) {
  std::pair<int, int> indexes = std::pair<int, int>(-1, -1);
  QPointF localPos = this->mapFromScene(point);
  if (_currentPath.elementCount() > 0) {
    QPointF prev(_currentPath.elementAt(0).x, _currentPath.elementAt(0).y);
    unsigned int lineIndex = 0;
    for (unsigned int el = 1; el < _currentPath.elementCount(); ++el) {
      QPainterPath::Element elm = _currentPath.elementAt(el);
      if (elm.type == 1) {
        QGraphicsLineItem line;
        line.setPen(QPen(QBrush(), _lineThickness / _currentLoD));
        line.setLine(prev.x(), prev.y(), elm.x, elm.y);
        if (line.contains(localPos)) {
          indexes = std::pair<int, int>(lineIndex, lineIndex + 1);
          break;
        }
        lineIndex += 1;
      }
      else if (elm.type == 2) {
        QPainterPath::Element cp2 = _currentPath.elementAt(el + 1);
        QPainterPath::Element ep = _currentPath.elementAt(el + 2);
        QPainterPath tmp(prev);
        tmp.cubicTo(QPointF(elm.x, elm.y), QPointF(cp2.x, cp2.y), QPointF(ep.x, ep.y));
        QPainterPathStroker stroker;
        stroker.setWidth(_lineThickness / _currentLoD);
        QGraphicsPathItem pth;
        pth.setPath(stroker.createStroke(tmp));
        if (pth.contains(localPos)) {
          indexes = std::pair<int, int>(lineIndex, lineIndex + 1);
          break;
        }
        lineIndex += 1;
      }
      prev = QPointF(elm.x, elm.y);
    }
  }
  return indexes;
}

void PolyQtAnnotation::finish() {
  prepareGeometryChange();
  _closed = true;
}

void PolyQtAnnotation::moveCoordinateBy(unsigned int index, const Point& moveBy) {
  std::vector<Point> coords = this->getAnnotation()->getCoordinates();
  if (index < coords.size()) {
    prepareGeometryChange();
    coords[index].setX(coords[index].getX() + moveBy.getX() / _scale);
    coords[index].setY(coords[index].getY() + moveBy.getY() / _scale);
    this->getAnnotation()->setCoordinates(coords);
    if (index == 0) {
      this->setPos(QPointF(coords[0].getX()*_scale, coords[0].getY()*_scale));
    }
  }
}

void PolyQtAnnotation::moveCoordinatesBy(const Point& moveBy) {
  std::vector<Point> coords = this->getAnnotation()->getCoordinates();
  for (std::vector<Point>::iterator it = coords.begin(); it != coords.end(); ++it) {
    it->setX(it->getX() + moveBy.getX()/_scale);
    it->setY(it->getY() + moveBy.getY()/_scale);
  }
  this->getAnnotation()->setCoordinates(coords);
  this->setPos(QPointF(coords[0].getX()*_scale, coords[0].getY()*_scale));
}

QPainterPath PolyQtAnnotation::shape() const {  
  QPainterPath rectPath;
  QPainterPathStroker stroker;
  QPainterPath strokePath;
  if (_annotation) {
    std::vector<Point> coords = _annotation->getCoordinates();
    for (unsigned int i = 0; i < coords.size(); ++i) {
      rectPath.addRect(QRectF(this->mapFromScene(coords[i].getX()*_scale - ((_rectSize + _lineThickness) / _currentLoD) / 2., coords[i].getY()*_scale - ((_rectSize + _lineThickness) / _currentLoD) / 2.), QSizeF((_rectSize + _lineThickness) / _currentLoD, (_rectSize + _lineThickness) / _currentLoD)));
    }
    if (isSelected()) {
      stroker.setWidth(_lineAnnotationSelectedThickness / _currentLoD);
    }
    else {
      stroker.setWidth(_lineThickness / _currentLoD);
    }
    strokePath = stroker.createStroke(_currentPath).subtracted(rectPath);
    for (unsigned int i = 0; i < coords.size(); ++i) {
      strokePath.addRect(QRectF(this->mapFromScene(coords[i].getX()*_scale - ((_rectSize + _lineThickness) / _currentLoD) / 2., coords[i].getY()*_scale - ((_rectSize + _lineThickness) / _currentLoD) / 2.), QSizeF((_rectSize + _lineThickness) / _currentLoD, (_rectSize + _lineThickness) / _currentLoD)));
    }
  }
  return strokePath.simplified();
}
