#ifndef ANNOTATIONTOOL_H
#define ANNOTATIONTOOL_H

#include "interfaces/interfaces.h"
#include "core/Point.h"
#include "annotationplugin_export.h"

class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;

class ANNOTATIONPLUGIN_EXPORT AnnotationTool : public  ToolPluginInterface {
  Q_OBJECT

public :
  AnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer);
  virtual std::string name() = 0;
  virtual void mouseMoveEvent(QMouseEvent *event);
  virtual void mousePressEvent(QMouseEvent *event);
  virtual void mouseReleaseEvent(QMouseEvent *event);
  virtual void mouseDoubleClickEvent(QMouseEvent *event);
  virtual void keyPressEvent(QKeyEvent *event);
  virtual QAction* getToolButton() = 0;
  void setActive(bool active);

public slots:
  virtual void cancelAnnotation();

protected slots:
  void zoomToAnnotation(qreal val);
  void zoomToAnnotationFinished();

protected:
  virtual void addCoordinate(const QPointF& scenePos);

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