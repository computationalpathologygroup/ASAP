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
  QAction* getToolButton();

private :
  AnnotationWorkstationExtensionPlugin* _annotationPlugin;
};

#endif