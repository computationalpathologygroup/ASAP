#ifndef POINTSETQTANNOTATION_H
#define POINTSETQTANNOTATION_H
#include "QtAnnotation.h"
#include <QColor>
#include <memory>
#include "annotationplugin_export.h"

class ANNOTATIONPLUGIN_EXPORT PointSetQtAnnotation : public QtAnnotation
{
  Q_OBJECT
public:
  PointSetQtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject *parent, float scale = 1.0);
  QRectF boundingRect() const;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
    QWidget *widget);

  bool contains(const QPointF & point) const;
  bool collidesWithPath(const QPainterPath & path, Qt::ItemSelectionMode mode = Qt::IntersectsItemShape) const;

private:
  QColor _rectColor;
  QColor _rectSelectedColor;
  float _currentLoD;
  float _rectSize;
  QRectF _bRect;

  void onAnnotationChanged();

};
#endif