#include "SplineAnnotationTool.h"
#include <QAction>
#include <QPixmap>
#include <QIcon>
#include "AnnotationWorkstationExtensionPlugin.h"
#include "../PathologyViewer.h"

SplineAnnotationTool::SplineAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, ASAP::PathologyViewController& controller) :
PolyAnnotationTool(annotationPlugin, controller)
{
}

QAction* SplineAnnotationTool::getToolButton() {
  if (!_button) {
    _button = new QAction("SplineAnnotation", this);
    _button->setObjectName(QString::fromStdString(name()));
    _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/spline.png")));
  }
  return _button;
}

std::string SplineAnnotationTool::name() {
  return std::string("splineannotation");
}