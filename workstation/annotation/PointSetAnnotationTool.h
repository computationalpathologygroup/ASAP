#ifndef POINTSETANNOTATIONTOOL_H
#define POINTSETANNOTATIONTOOL_H

#include "PolyAnnotationTool.h"

class QAction;
class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;

class PointSetAnnotationTool : public PolyAnnotationTool {
  Q_OBJECT

public :
  PointSetAnnotationTool(AnnotationWorkstationExtensionPlugin *annotationPlugin, PathologyViewer *viewer);
  virtual std::string name();
  virtual QAction* getToolButton();
};

#endif