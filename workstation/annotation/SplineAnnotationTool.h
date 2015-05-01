#ifndef SPLINEANNOTATIONTOOL_H
#define SPLINEANNOTATIONTOOL_H

#include "PolyAnnotationTool.h"

class QAction;
class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;

class SplineAnnotationTool : public PolyAnnotationTool {
  Q_OBJECT

public :
  SplineAnnotationTool(AnnotationWorkstationExtensionPlugin *annotationPlugin, PathologyViewer *viewer);
  virtual std::string name();
  virtual QAction* getToolButton();
};

#endif