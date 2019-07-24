#ifndef DOTANNOTATIONTOOL_H
#define DOTANNOTATIONTOOL_H

#include "AnnotationTool.h"

class AnnotationWorkstationExtensionPlugin;
class PathologyViewer;

class ANNOTATIONPLUGIN_EXPORT DotAnnotationTool : public  AnnotationTool {
  Q_OBJECT

public :
  DotAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, ASAP::PathologyViewController& controller);
  std::string name();
  void mousePressEvent(QMouseEvent *event);

  QAction* getToolButton();

};

#endif