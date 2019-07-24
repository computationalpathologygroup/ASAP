#ifndef SPLINEANNOTATIONTOOL_H
#define SPLINEANNOTATIONTOOL_H

#include "PolyAnnotationTool.h"

class QAction;
class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;

class ANNOTATIONPLUGIN_EXPORT SplineAnnotationTool : public PolyAnnotationTool {
  Q_OBJECT

public :
  SplineAnnotationTool(AnnotationWorkstationExtensionPlugin *annotationPlugin, ASAP::PathologyViewController& controller);
  virtual std::string name();
  virtual QAction* getToolButton();
};

#endif