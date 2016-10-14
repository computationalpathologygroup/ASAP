#ifndef MEASUREMENTANNOTATIONTOOL_H
#define MEASUREMENTANNOTATIONTOOL_H

#include "AnnotationTool.h"
#include "core/Point.h"
#include <QGraphicsTextItem>

class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;
class QGraphicsLineItem;

class ANNOTATIONPLUGIN_EXPORT MeasurementAnnotationTool : public AnnotationTool {
  Q_OBJECT

public :
  MeasurementAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer);
  virtual std::string name();
  virtual QAction* getToolButton();
  void mouseMoveEvent(QMouseEvent *event);
  void keyPressEvent(QKeyEvent *event);
  void cancelAnnotation();

private:

  class QGraphicsTextItemWithBackground : public QGraphicsTextItem
  {
  public:
    QGraphicsTextItemWithBackground(const QString &text);
    QRectF boundingRect() const;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *o, QWidget *w);
  };

  void addCoordinate(const QPointF& scenePos);
  QGraphicsLineItem* _activeLine;
  QGraphicsTextItemWithBackground* _sizeText;
};

#endif