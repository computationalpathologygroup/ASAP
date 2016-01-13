#ifndef DOTQTANNOTATION_H
#define DOTQTANNOTATION_H
#include <QColor>
#include "QtAnnotation.h"
#include "config/pathology_config.h"

class EXPORT_PATHOLOGYANNOTATIONPLUGIN DotQtAnnotation : public QtAnnotation
{
  Q_OBJECT
public:
  DotQtAnnotation(const std::shared_ptr<Annotation>& annotation, float scale = 1.0);
  QRectF boundingRect() const;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
    QWidget *widget);

private:
  float _rectSize;
  float _currentLOD;
  QColor _rectColor;
  QColor _rectSelectedColor;
};
#endif