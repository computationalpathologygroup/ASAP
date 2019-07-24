#include "DotAnnotationTool.h"
#include "DotQtAnnotation.h"
#include <QAction>
#include "../PathologyViewer.h"
#include "AnnotationWorkstationExtensionPlugin.h"
#include "core/Point.h"

DotAnnotationTool::DotAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, ASAP::PathologyViewController& controller) :
  AnnotationTool(annotationPlugin, controller)
{
}

void DotAnnotationTool::mousePressEvent(QMouseEvent *event) {
  AnnotationTool::mousePressEvent(event);
  if (_generating) {
    _annotationPlugin->finishAnnotation();
    _start = Point(-1, -1);
    _last = _start;
    _generating = false;
  }
  event->accept();
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