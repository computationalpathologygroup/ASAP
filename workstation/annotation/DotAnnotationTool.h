#ifndef DOTANNOTATIONTOOL_H
#define DOTANNOTATIONTOOL_H

#include "AnnotationTool.h"

class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;

class DotAnnotationTool : public  AnnotationTool {
  Q_OBJECT

public :
  DotAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer);
  std::string name();
  void mousePressEvent(QMouseEvent *event);

  QAction* getToolButton();

};

#endif