#include "DotAnnotationTool.h"
#include <QAction>
#include "../PathologyViewer.h"
#include "AnnotationWorkstationExtensionPlugin.h"

DotAnnotationTool::DotAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) : 
  ToolPluginInterface(),
  _annotationPlugin(annotationPlugin)
{
  _viewer = viewer;
}

void DotAnnotationTool::mouseMoveEvent(QMouseEvent *event) {
  if (_viewer) {
    event->accept();
  }
}

void DotAnnotationTool::mousePressEvent(QMouseEvent *event) {
  if (_viewer) {
    QPointF scenePos = _viewer->mapToScene(event->pos());
    _annotationPlugin->startAnnotation(scenePos.x(), scenePos.y(), name());
    _annotationPlugin->finishAnnotation();
    event->accept();
  }
}

void DotAnnotationTool::mouseReleaseEvent(QMouseEvent *event) {
  if (_viewer) {
    event->accept();
  }
}

QAction* DotAnnotationTool::getToolButton() {
  if (!_button) {
    _button = new QAction("DotAnnotation", this);
    _button->setObjectName(QString::fromStdString(name()));
    _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/dot.png")));
  }
  return _button;
}

std::string DotAnnotationTool::name() {
  return std::string("dotannotation");
}