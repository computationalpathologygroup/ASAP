#ifndef DOTANNOTATIONTOOL_H
#define DOTANNOTATIONTOOL_H

#include "../interfaces/interfaces.h"

class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;

class DotAnnotationTool : public  ToolPluginInterface {
  Q_OBJECT

public :
  DotAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer);
  std::string name();
  void mouseMoveEvent(QMouseEvent *event);
  void mousePressEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  void keyPressEvent(QKeyEvent *event);

  QAction* getToolButton();

private :
  AnnotationWorkstationExtensionPlugin* _annotationPlugin;
  bool _startSelectionMove;
  QPointF _moveStart;
};

#endif