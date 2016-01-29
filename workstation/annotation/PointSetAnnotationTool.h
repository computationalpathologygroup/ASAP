#ifndef POINTSETANNOTATIONTOOL_H
#define POINTSETANNOTATIONTOOL_H

#include "PolyAnnotationTool.h"

class QAction;
class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;

class PointSetAnnotationTool : public  ToolPluginInterface {
  Q_OBJECT

public:
  PointSetAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer);
  virtual std::string name();
  void mouseMoveEvent(QMouseEvent *event);
  void mousePressEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  void mouseDoubleClickEvent(QMouseEvent *event);
  void keyPressEvent(QKeyEvent *event);
  virtual QAction* getToolButton();
  void setActive(bool active);

  public slots:
  void cancelAnnotation();

  private slots:
  void zoomToAnnotation(qreal val);
  void zoomToAnnotationFinished();

private:

  void addCoordinate(const QPointF& scenePos);

  AnnotationWorkstationExtensionPlugin* _annotationPlugin;
  bool _generating;
  Point _start;
  Point _last;

  bool _startSelectionMove;
  QPointF _moveStart;
  QRectF _start_zoom;
  QRectF _end_zoom;
};

#endif