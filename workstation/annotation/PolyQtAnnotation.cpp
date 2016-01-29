#include "PolyQtAnnotation.h"
#include "Annotation.h"
#include <QPainter>
#include <QPainterPath>
#include <QPainterPathStroker>
#include <QStyleOptionGraphicsItem>
#include <iostream>

PolyQtAnnotation::PolyQtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject* parent, float scale) : 
  QtAnnotation(annotation, parent, scale),
  _lineThickness(3),
  _lineAnnotationSelectedThickness(4.5),
  _rectColor(QColor("blue")),
  _rectSelectedColor(QColor("red")),
  _closed(false),
  _type("spline"),
  _currentLoD(1.0),
  _selectionSensitivity(100.0),
  _lastClickedLinePoint(QPointF()),
  _lastClickedFirstCoordinateIndex(-1),
  _lastClickedSecondCoordinateIndex(-1),
  _fill(false)
{

}

QRectF PolyQtAnnotation::boundingRect() const {
  QRectF bRect;
  if (_annotation) {
      QRectF cpRect = _currentPath.controlPointRect();
      QPointF tl = cpRect.topLeft() - QPointF(3 * _lineAnnotationSelectedThickness, 3 * _lineAnnotationSelectedThickness);
      QPointF br = cpRect.bottomRight() + QPointF(3 * _lineAnnotationSelectedThickness, 3 * _lineAnnotationSelectedThickness);
      bRect = bRect.united(QRectF(tl, br));
  }
  return bRect;
}

void PolyQtAnnotation::onCoordinatesChanged() {
  _currentPath = getCurrentPath(_annotation->getCoordinates());
  if (_type == "spline") {
    _polys = _currentPath.toFillPolygon();
    if (!_closed && _polys.isClosed()) {
      _polys.pop_back();
    }
  }
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
  if (coords.size() > 1) {
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
        if (coords.size() > 1) {
          QPointF p1 = this->mapFromScene(coords[coords.size() - 1].getX()*_scale, coords[coords.size() - 1].getY()*_scale);
          QPointF p2 = this->mapFromScene(coords[0].getX()*_scale, coords[0].getY()*_scale);
          QPointF p0 = p1 - (p2 - p1);
          if (coords.size() > 2) {
            QPointF p0 = this->mapFromScene(coords[coords.size() - 2].getX()*_scale, coords[coords.size() - 2].getY()*_scale);
          }
          QPointF p3 = this->mapFromScene(coords[1].getX()*_scale, coords[1].getY()*_scale);
          std::vector<QPointF> bezierPoints = catmullRomToBezier(p0, p1, p2, p3);
          pth.cubicTo(bezierPoints[1], bezierPoints[2], bezierPoints[3]);
        }
      }
    }
  }
  return pth;
}

void PolyQtAnnotation::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
  QWidget *widget) {
  if (_annotation) {
    QColor lineColor = this->getDrawingColor();
    QColor fillColor = this->getDrawingColor();
    fillColor.setAlphaF(0.3);
    _currentLoD = option->levelOfDetailFromTransform(painter->worldTransform());
    std::vector<Point> coords = _annotation->getCoordinates();
    if (coords.size() > 1) {
      if (isSelected()) {
        painter->setPen(QPen(QBrush(lineColor.lighter(150)), _lineAnnotationSelectedThickness / _currentLoD));
      }
      else {
        painter->setPen(QPen(QBrush(lineColor), _lineThickness / _currentLoD));
      }
      if (_type == "spline") {
        painter->drawPolyline(_polys);
        if (_fill) {
          QPainterPath path;
          path.addPolygon(_polys);
          painter->fillPath(path, QBrush(fillColor));
        }
      }
      else {
        painter->drawPath(_currentPath);
        if (_fill) {
          painter->fillPath(_currentPath, QBrush(fillColor));
        }
      }
    }
    if (isSelected()) {
      painter->setPen(QPen(QBrush(_rectColor.lighter(150)), 3 * _lineThickness / _currentLoD, Qt::PenStyle::SolidLine, Qt::PenCapStyle::SquareCap));
    }
    else {
      painter->setPen(QPen(QBrush(_rectColor), 3 * _lineThickness / _currentLoD, Qt::PenStyle::SolidLine, Qt::PenCapStyle::SquareCap));
    }
    for (unsigned int i = 0; i < coords.size(); ++i) {
      if (i == _activeSeedPoint) {
        painter->save();
        painter->setPen(QPen(QBrush(_rectSelectedColor), 3 * _lineAnnotationSelectedThickness / _currentLoD, Qt::PenStyle::SolidLine, Qt::PenCapStyle::SquareCap));
        painter->drawPoint(this->mapFromScene(coords[i].getX()*_scale, coords[i].getY()*_scale));
        painter->restore();
      }
      else {
        painter->drawPoint(this->mapFromScene(coords[i].getX()*_scale, coords[i].getY()*_scale));
      }
    }
  }
}

void PolyQtAnnotation::finish() {
  prepareGeometryChange();
  _closed = true;
  onCoordinatesChanged();
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
  onCoordinatesChanged();
}

void PolyQtAnnotation::moveCoordinatesBy(const Point& moveBy) {
  std::vector<Point> coords = this->getAnnotation()->getCoordinates();
  for (std::vector<Point>::iterator it = coords.begin(); it != coords.end(); ++it) {
    it->setX(it->getX() + moveBy.getX()/_scale);
    it->setY(it->getY() + moveBy.getY()/_scale);
  }
  this->getAnnotation()->setCoordinates(coords);
  this->setPos(QPointF(coords[0].getX()*_scale, coords[0].getY()*_scale));
  onCoordinatesChanged();
}

bool PolyQtAnnotation::collidesWithPath(const QPainterPath & path, Qt::ItemSelectionMode mode) const {
  return contains(path.currentPosition());
}

bool PolyQtAnnotation::contains(const QPointF & point) const {
  if (shape().controlPointRect().contains(point)) {
    QPointF imgPoint = this->mapToScene(point) / _scale;
    double curSelectionSensitivity = (_selectionSensitivity * _lineAnnotationSelectedThickness / _currentLoD);
    double curSelectionSensitivitySquared = curSelectionSensitivity * curSelectionSensitivity;
    double imgX = imgPoint.x();
    double imgY = imgPoint.y();
    std::vector<Point> coords = _annotation->getCoordinates();
    double minDist = std::numeric_limits<double>::max();
    _lastClickedFirstCoordinateIndex = -1;
    _lastClickedSecondCoordinateIndex = -1;

    // Quickly check if a seed point was hit
    for (unsigned int i = 0; i < coords.size(); ++i) {
      Point pt1 = coords[i];
      double coord1X = pt1.getX(); double coord1Y = pt1.getY();
      double distSquared = pow(imgX - coord1X, 2) + pow(imgY - coord1Y, 2);
      if (distSquared < curSelectionSensitivitySquared && distSquared < minDist) {
        _lastClickedFirstCoordinateIndex = i;
        _lastClickedSecondCoordinateIndex = -1;
        _lastClickedLinePoint = QPointF();
        minDist = distSquared;
      }
    }
    if (_lastClickedFirstCoordinateIndex >= 0) {
      return true;
    }

    minDist = std::numeric_limits<double>::max();
    // If not, check if a line was hit
    std::vector<QPointF> polyInImgCoords;
    unsigned int polyIndex = 0;
    if (_type == "spline") {
      for (QPolygonF::const_iterator it = _polys.begin(); it != _polys.end(); ++it) {
        polyInImgCoords.push_back(this->mapToScene(*it) / _scale);
      }
    }
    for (unsigned int i = 0; i < coords.size(); ++i) {
      Point pt1 = coords[i];
      Point pt2 = i == coords.size() -1 ? coords[0] : coords[i + 1];
      double coord1X = pt1.getX(); double coord1Y = pt1.getY();
      double coord2X = pt2.getX(); double coord2Y = pt2.getY();
      QRectF hitbox(imgX - curSelectionSensitivity / 2., imgY - curSelectionSensitivity / 2., curSelectionSensitivity * 2., curSelectionSensitivity * 2.);
      QRectF lineBox(QPointF(std::min(coord1X, coord2X), std::max(coord1Y, coord2Y)), QPointF(std::max(coord1X, coord2X), std::min(coord1Y, coord2Y)));       
      if (hitbox.intersects(lineBox)) {
        if (_type == "spline") {
          for (unsigned int j = 0; j < polyInImgCoords.size(); ++j) {
            QPointF polyPt1 = polyInImgCoords[polyIndex];
            QPointF polyPt2 = polyIndex == polyInImgCoords.size() - 1 ? polyInImgCoords[0] : polyInImgCoords[polyIndex + 1];
            if (QPoint(polyPt1.x(), polyPt1.y()) == QPoint(coord2X, coord2Y)) {
              break;
            }
            double polyCoord1X = polyPt1.x(); double polyCoord1Y = polyPt1.y();
            double polyCoord2X = polyPt2.x(); double polyCoord2Y = polyPt2.y();
            QRectF polyBox(QPointF(std::min(polyCoord1X, polyCoord2X), std::max(polyCoord1Y, polyCoord2Y)), QPointF(std::max(polyCoord1X, polyCoord2X), std::min(polyCoord1Y, polyCoord2Y)));
            if (hitbox.intersects(polyBox)) {
              double lineLengthSquared = pow(polyCoord1X - polyCoord2X, 2) + pow(polyCoord1Y - polyCoord2Y, 2);
              double t = ((imgX - polyCoord2X) * (polyCoord1X - polyCoord2X) + (imgY - polyCoord2Y) * (polyCoord1Y - polyCoord2Y)) / lineLengthSquared;
              double projX = polyCoord2X + t * (polyCoord1X - polyCoord2X);
              double projY = polyCoord2Y + t * (polyCoord1Y - polyCoord2Y);
              double distSquared = pow(imgX - projX, 2) + pow(imgY - projY, 2);
              if (distSquared < curSelectionSensitivitySquared && distSquared < minDist) {
                _lastClickedFirstCoordinateIndex = i;
                _lastClickedSecondCoordinateIndex = i + 1 == coords.size() ? 0 : i + 1;
                _lastClickedLinePoint = QPointF(projX, projY);
              }
            }
            ++polyIndex;
          }
        }
        else {
          double lineLengthSquared = pow(coord1X - coord2X, 2) + pow(coord1Y - coord2Y, 2);
          double t = ((imgX - coord2X) * (coord1X - coord2X) + (imgY - coord2Y) * (coord1Y - coord2Y)) / lineLengthSquared;
          double projX = coord2X + t * (coord1X - coord2X);
          double projY = coord2Y + t * (coord1Y - coord2Y);
          double distSquared = pow(imgX - projX, 2) + pow(imgY - projY, 2);
          if (distSquared < curSelectionSensitivitySquared && distSquared < minDist) {
            _lastClickedFirstCoordinateIndex = i;
            _lastClickedSecondCoordinateIndex = i + 1 == coords.size() ? 0 : i + 1;
            _lastClickedLinePoint = QPointF(projX, projY);
          }
        }
      }
    }
    if (_lastClickedFirstCoordinateIndex < 0) {
      return false;
    }
    else {
      return true;
    }
  }
  return false;
}

QPointF PolyQtAnnotation::getLastClickedLinePoint() {
  return _lastClickedLinePoint;
}

std::pair<int, int> PolyQtAnnotation::getLastClickedCoordinateIndices() {
  return std::pair<int, int>(_lastClickedFirstCoordinateIndex, _lastClickedSecondCoordinateIndex);
}