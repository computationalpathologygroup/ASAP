#include "DotQtAnnotation.h"
#include "Annotation.h"
#include <QPainter>
#include <QStyleOptionGraphicsItem>

DotQtAnnotation::DotQtAnnotation(const std::shared_ptr<Annotation>& annotation, float scale) : 
  QtAnnotation(annotation, scale),
  _rectSize(10),
  _rectColor(QColor("blue"))
{

}

QRectF DotQtAnnotation::boundingRect() const {
  QRectF bRect;
  if (_annotation) {
    std::vector<Point> bbox = _annotation->getLocalBoundingBox();
    QPointF tl((_scale*bbox[0].getX()) - _rectSize, (_scale*bbox[0].getY()) - _rectSize);
    QPointF br((_scale*bbox[1].getX()) + _rectSize, (_scale*bbox[1].getY()) + _rectSize);
    bRect = QRectF(tl, br);
  }
  return bRect;
}

void DotQtAnnotation::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
  QWidget *widget) {
  if (_annotation) {
    _rectColor = this->getDrawingColor();
    float lod = option->levelOfDetailFromTransform(painter->worldTransform());
    std::vector<Point> coords = _annotation->getCoordinates();
    for (std::vector<Point>::const_iterator it = coords.begin(); it != coords.end(); ++it) {
      painter->setPen(QPen(QBrush(_rectColor), 3 / lod));
      painter->drawRect(QRectF(QPointF(-(_rectSize/lod)/2., -(_rectSize/lod)/2.), QSizeF(_rectSize/lod, _rectSize/lod)));
    }
  }
}