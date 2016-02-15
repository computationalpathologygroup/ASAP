#ifndef POINTSETANNOTATIONTOOL_H
#define POINTSETANNOTATIONTOOL_H

#include "AnnotationTool.h"

class QAction;
class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;

class PointSetAnnotationTool : public  AnnotationTool {
  Q_OBJECT

public:
  
  PointSetAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer);
  std::string name();
  void mouseDoubleClickEvent(QMouseEvent *event);
  QAction* getToolButton();

};

#endif