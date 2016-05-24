#ifndef POLYANNOTATIONTOOL_H
#define POLYANNOTATIONTOOL_H

#include "AnnotationTool.h"
#include "core/Point.h"

class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;
class QGraphicsLineItem;

class ANNOTATIONPLUGIN_EXPORT PolyAnnotationTool : public AnnotationTool {
  Q_OBJECT

public :
  PolyAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer);
  virtual std::string name();
  void mouseMoveEvent(QMouseEvent *event);
  void mouseDoubleClickEvent(QMouseEvent *event);
  void cancelAnnotation();
  virtual QAction* getToolButton();

private:
  void addCoordinate(const QPointF& scenePos);
  QGraphicsLineItem* _activeLine;
};

#endif