#ifndef MEASUREMENTQTANNOTATION_H
#define MEASUREMENTQTANNOTATION_H
#include "QtAnnotation.h"
#include <QColor>
#include <memory>
#include "annotationplugin_export.h"

class ANNOTATIONPLUGIN_EXPORT MeasurementQtAnnotation : public QtAnnotation
{
  Q_OBJECT
public:
  MeasurementQtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject *parent, float scale = 1.0);
  QRectF boundingRect() const;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
    QWidget *widget);

  bool contains(const QPointF & point) const;
  bool collidesWithPath(const QPainterPath & path, Qt::ItemSelectionMode mode = Qt::IntersectsItemShape) const;

private:
  float _lineThickness;
  float _lineAnnotationSelectedThickness;
  float _selectionSensitivity;
  float _currentLoD;
  std::vector<double> _spacing;
};
#endif