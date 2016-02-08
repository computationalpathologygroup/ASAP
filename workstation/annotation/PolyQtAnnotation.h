#ifndef POLYQTANNOTATION_H
#define POLYQTANNOTATION_H
#include "QtAnnotation.h"
#include <QColor>
#include <memory>
#include "config/pathology_config.h"

class EXPORT_PATHOLOGYANNOTATIONPLUGIN PolyQtAnnotation : public QtAnnotation
{
  Q_OBJECT
public:
  PolyQtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject *parent, float scale = 1.0);
  QRectF boundingRect() const;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
    QWidget *widget);

  void finish();
  void setInterpolationType(const std::string& interpolationType);
  std::string getInterpolationType();
  QPainterPath getCurrentPath(const std::vector<Point>& coords) const;
  bool contains(const QPointF & point) const;
  bool collidesWithPath(const QPainterPath & path, Qt::ItemSelectionMode mode = Qt::IntersectsItemShape) const;
  QPointF getLastClickedLinePoint();
  std::pair<int, int> getLastClickedCoordinateIndices();

private:
  QColor _rectColor;
  QColor _rectSelectedColor;
  float _lineThickness;
  float _lineAnnotationSelectedThickness;
  bool _closed;
  float _selectionSensitivity;
  QPainterPath _currentPath;
  QPolygonF _polys;
  std::string _type;
  float _currentLoD;
  bool _fill;
  mutable QPointF _lastClickedLinePoint;
  mutable int _lastClickedFirstCoordinateIndex;
  mutable int _lastClickedSecondCoordinateIndex;

  std::vector<QPointF> catmullRomToBezier(const QPointF& p0, const QPointF& p1, const QPointF& p2, const QPointF& p3) const;

  void onCoordinatesChanged();

};
#endif