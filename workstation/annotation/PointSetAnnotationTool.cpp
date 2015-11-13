#include "PointSetAnnotationTool.h"
#include <QAction>
#include <QPixmap>
#include <QIcon>
#include "AnnotationWorkstationExtensionPlugin.h"
#include "../PathologyViewer.h"

PointSetAnnotationTool::PointSetAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) :
PolyAnnotationTool(annotationPlugin, viewer)
{
}

QAction* PointSetAnnotationTool::getToolButton() {
  if (!_button) {
    _button = new QAction("PointSetAnnotation", this);
    _button->setObjectName(QString::fromStdString(name()));
    _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/pointset.png")));
  }
  return _button;
}

std::string PointSetAnnotationTool::name() {
  return std::string("pointsetannotation");
}