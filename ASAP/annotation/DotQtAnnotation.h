#ifndef DOTQTANNOTATION_H
#define DOTQTANNOTATION_H
#include <QColor>
#include "QtAnnotation.h"
#include "annotationplugin_export.h"

class ANNOTATIONPLUGIN_EXPORT DotQtAnnotation : public QtAnnotation
{
  Q_OBJECT
public:
  DotQtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject* parent, float scale = 1.0);
  QRectF boundingRect() const;
  QPainterPath shape() const;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
    QWidget *widget);

  void moveCoordinateBy(const Point& moveBy);
  void moveCoordinatesBy(const Point& moveBy);

private:
  float _rectSize;
  float _currentLOD;
  QColor _rectColor;
  QColor _rectSelectedColor;
};
#endif