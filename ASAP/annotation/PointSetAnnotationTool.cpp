#include "PointSetAnnotationTool.h"
#include "QtAnnotation.h"
#include "AnnotationWorkstationExtensionPlugin.h"
#include <QAction>
#include <QPen>
#include <QGraphicsLineItem>
#include <QGraphicsItem>
#include "PointSetQtAnnotation.h"
#include "annotation/Annotation.h"
#include "../PathologyViewer.h"
#include <math.h>
#include <numeric>
#include <iostream>
#include <QTimeLine>

PointSetAnnotationTool::PointSetAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) :
AnnotationTool(annotationPlugin, viewer)
{
}

void PointSetAnnotationTool::mouseDoubleClickEvent(QMouseEvent *event) {
  if (_generating) {
    _annotationPlugin->finishAnnotation();
    _start = Point(-1, -1);
    _last = _start;
    _generating = false;
  }
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