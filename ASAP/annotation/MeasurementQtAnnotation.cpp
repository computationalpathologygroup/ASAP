#include "MeasurementQtAnnotation.h"
#include "annotation/Annotation.h"
#include <QObject>
#include <QPainter>
#include <QPainterPath>
#include <QPainterPathStroker>
#include <QStyleOptionGraphicsItem>
#include "AnnotationWorkstationExtensionPlugin.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include <iostream>
#include <cmath>

MeasurementQtAnnotation::MeasurementQtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject* parent, float scale) :
  QtAnnotation(annotation, parent, scale),
  _lineThickness(3),
  _lineAnnotationSelectedThickness(4.5),
  _currentLoD(1.0),
  _selectionSensitivity(100.0)
{
  annotation->setColor("#000000");
  AnnotationWorkstationExtensionPlugin* annot_plugin = dynamic_cast<AnnotationWorkstationExtensionPlugin*>(parent);
  if (annot_plugin) {
    if (std::shared_ptr<MultiResolutionImage> local_img = annot_plugin->getCurrentImage().lock()) {
      _spacing = local_img->getSpacing();
    }
  }
}

QRectF MeasurementQtAnnotation::boundingRect() const {
  QRectF bRect;
  if (_annotation) {    
    std::vector<Point> coords = _annotation->getCoordinates();
    QPointF last = this->mapFromScene(coords[coords.size() - 1].getX()*_scale, coords[coords.size() - 1].getY()*_scale);
    float left = last.x() <= 0 ? last.x() : 0;
    float top = last.y() <= 0 ? last.y() : 0;
    float right = last.x() > 0 ? last.x() : 0;
    float bottom = last.y() > 0 ? last.y() : 0;
    bRect = QRectF(left, top, right - left, bottom - top);
  }
  return bRect;
}

void MeasurementQtAnnotation::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
  QWidget *widget) {
  if (_annotation) {
    QColor lineColor = this->getDrawingColor();
    _currentLoD = option->levelOfDetailFromTransform(painter->worldTransform());
    std::vector<Point> coords = _annotation->getCoordinates();
    if (coords.size() > 1) {
      if (isSelected()) {
        painter->setPen(QPen(QBrush(lineColor.lighter(150)), _lineAnnotationSelectedThickness / _currentLoD));
      }
      else {
        painter->setPen(QPen(QBrush(lineColor), _lineThickness / _currentLoD));
      }
      painter->drawLine(QPointF(0, 0), this->mapFromScene(coords[coords.size() - 1].getX()*_scale, coords[coords.size() - 1].getY()*_scale));
      Point firstPoint = coords[0];
      Point lastPoint = coords[coords.size() - 1];
      QPointF centerPoint = this->mapFromScene(lastPoint.getX()*_scale, lastPoint.getY()*_scale) / 2;
      float vecLength = sqrt(QPointF::dotProduct(centerPoint, centerPoint));
      QPointF unitVec = QPointF(centerPoint.y(), -centerPoint.x()) / vecLength;
      QPointF textPos = centerPoint.y() < 0 ? (centerPoint - (15 / _currentLoD) * unitVec) * _currentLoD : (centerPoint + (15 / _currentLoD) * unitVec) * _currentLoD;;
      QPainterPath textPath;
      QFont ft = QFont("Arial");
      float sizeInPixels = sqrt(pow(firstPoint.getX() - lastPoint.getX(), 2) + pow(firstPoint.getY() - lastPoint.getY(), 2));
      if (_spacing.size() > 0) {
        float sizeInUnits = sizeInPixels * _spacing[0];
        QString unit = " um";
        if (sizeInUnits > 1000) {
          sizeInUnits /= 1000;
          unit = " mm";
        }
        textPath.addText(textPos, ft, QString::number(sizeInUnits, 'g', 4) + unit);
      }
      else {        
        textPath.addText(textPos, ft, QString::number(sizeInPixels) + QString(" pixels"));
      }
      painter->scale(1.5 / _currentLoD, 1.5 / _currentLoD);
      painter->setPen(Qt::NoPen);
      painter->setBrush(QBrush(QColor(0,0,0,75)));
      painter->drawRect(textPath.boundingRect().adjusted(-5,-5,5,5));
      painter->setBrush(QBrush(Qt::white));
      painter->drawPath(textPath);
    }
  }
}

bool MeasurementQtAnnotation::collidesWithPath(const QPainterPath & path, Qt::ItemSelectionMode mode) const {
  return contains(path.currentPosition());
}

bool MeasurementQtAnnotation::contains(const QPointF & point) const {
  if (shape().controlPointRect().contains(point)) {
    QPointF imgPoint = this->mapToScene(point) / _scale;
    double curSelectionSensitivity = (_selectionSensitivity * _lineAnnotationSelectedThickness / _currentLoD);
    double curSelectionSensitivitySquared = curSelectionSensitivity * curSelectionSensitivity;
    double imgX = imgPoint.x();
    double imgY = imgPoint.y();
    std::vector<Point> coords = _annotation->getCoordinates();
    double minDist = std::numeric_limits<double>::max();

    minDist = std::numeric_limits<double>::max();
    // If not, check if a line was hit
    for (unsigned int i = 0; i < coords.size(); ++i) {
      Point pt1 = coords[i];
      Point pt2 = i == coords.size() -1 ? coords[0] : coords[i + 1];
      double coord1X = pt1.getX(); double coord1Y = pt1.getY();
      double coord2X = pt2.getX(); double coord2Y = pt2.getY();
      QRectF hitbox(imgX - curSelectionSensitivity / 2., imgY - curSelectionSensitivity / 2., curSelectionSensitivity * 2., curSelectionSensitivity * 2.);
      QRectF lineBox(QPointF(std::min(coord1X, coord2X) - curSelectionSensitivity / 2., std::max(coord1Y, coord2Y) + curSelectionSensitivity / 2.), QPointF(std::max(coord1X, coord2X) + curSelectionSensitivity / 2., std::min(coord1Y, coord2Y) - curSelectionSensitivity / 2.));
      if (hitbox.intersects(lineBox)) {
        double lineLengthSquared = pow(coord1X - coord2X, 2) + pow(coord1Y - coord2Y, 2);
        double t = ((imgX - coord2X) * (coord1X - coord2X) + (imgY - coord2Y) * (coord1Y - coord2Y)) / lineLengthSquared;
        double projX = coord2X + t * (coord1X - coord2X);
        double projY = coord2Y + t * (coord1Y - coord2Y);
        double distSquared = pow(imgX - projX, 2) + pow(imgY - projY, 2);
        if (distSquared < curSelectionSensitivitySquared && distSquared < minDist) {
          return true;
        }
      }
    }
  }
  return false;
}