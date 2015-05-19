#ifndef POLYANNOTATIONTOOL_H
#define POLYANNOTATIONTOOL_H

#include "../interfaces/interfaces.h"
#include "core/Point.h"

class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;
class QGraphicsLineItem;

class PolyAnnotationTool : public  ToolPluginInterface {
  Q_OBJECT

public :
  PolyAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer);
  virtual std::string name();
  void mouseMoveEvent(QMouseEvent *event);
  void mousePressEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  void mouseDoubleClickEvent(QMouseEvent *event);
  void keyPressEvent(QKeyEvent *event);
  virtual QAction* getToolButton();

private :

  void addCoordinate(const QPointF& scenePos);
  void cancelAnnotation();

  AnnotationWorkstationExtensionPlugin* _annotationPlugin;
  bool _generating;
  Point _start;
  Point _last;
  QGraphicsLineItem* _activeLine;

  bool _startSelectionMove;
  QPointF _moveStart;
};

#endif