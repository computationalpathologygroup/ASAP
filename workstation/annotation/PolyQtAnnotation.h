#ifndef POLYQTANNOTATION_H
#define POLYQTANNOTATION_H
#include "QtAnnotation.h"
#include <QColor>
#include "config/pathology_config.h"

class EXPORT_PATHOLOGYANNOTATION PolyQtAnnotation : public QtAnnotation
{
  Q_OBJECT
public:
  PolyQtAnnotation(Annotation* annotation, float scale = 1.0);
  QRectF boundingRect() const;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
    QWidget *widget);

  void finish();
  QPainterPath shape() const;
  void moveCoordinateBy(unsigned int index, const Point& moveBy);
  void moveCoordinatesBy(const Point& moveBy);
  void setInterpolationType(const std::string& interpolationType);
  std::string getInterpolationType();
  std::pair<int, int> seedPointsContaininPathPoint(const QPointF& point);

private:
  float _rectSize;
  QColor _rectColor;
  QColor _rectSelectedColor;
  float _lineThickness;
  float _lineAnnotationSelectedThickness;
  QColor _lineColor;
  bool _closed;
  QPainterPath _currentPath;
  std::string _type;
  float _currentLoD;

  std::vector<QPointF> catmullRomToBezier(const QPointF& p0, const QPointF& p1, const QPointF& p2, const QPointF& p3);

};
#endif