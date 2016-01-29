#ifndef QTANNOTATION_H
#define QTANNOTATION_H
#include <string>
#include <vector>
#include <memory>
#include <QGraphicsItem>

#include "config/pathology_config.h"

class Annotation;
class Point;

class EXPORT_PATHOLOGYANNOTATIONPLUGIN QtAnnotation : public QObject, public QGraphicsItem
{
  Q_OBJECT
  Q_INTERFACES(QGraphicsItem)
public:
  QtAnnotation(const std::shared_ptr<Annotation>& annotation, QObject* parent, float scale = 1.);
  virtual ~QtAnnotation();
  void addCoordinate(const float& x, const float& y);
  void addCoordinate(const Point& xy);
  void insertCoordinate(const int& index, const float& x, const float& y);
  void insertCoordinate(const int& index, const Point& xy);
  void removeCoordinate(const int& index);
  void setCoordinates(const std::vector<Point>& coordinates);
  virtual void moveCoordinatesBy(const Point& moveBy) {};
	std::shared_ptr<Annotation> getAnnotation() const;
  
  virtual QRectF boundingRect() const = 0;

  virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
    QWidget *widget) = 0;

  virtual void finish() {};

  void setActiveSeedPoint(const unsigned int seedPointIndex);
  void clearActiveSeedPoint();
  int getActiveSeedPoint();

  void setEditable(const bool editable);
  bool getEditable() const;

  QColor getDrawingColor();

protected:
	std::shared_ptr<Annotation> _annotation;
  float _scale;
  int _activeSeedPoint;
  bool _editable;

  virtual void onCoordinatesChanged();

};

QDataStream &operator<<(QDataStream &out, QtAnnotation* const &rhs);
QDataStream &operator>>(QDataStream &in, QtAnnotation* &rhs);
#endif