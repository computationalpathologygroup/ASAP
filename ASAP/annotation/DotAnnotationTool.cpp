#include "DotAnnotationTool.h"
#include "DotQtAnnotation.h"
#include <QAction>
#include <QApplication>
#include <QStyleHints>
#include "../PathologyViewer.h"
#include "AnnotationWorkstationExtensionPlugin.h"
#include "core/Point.h"

DotAnnotationTool::DotAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) : 
  AnnotationTool(annotationPlugin, viewer)
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
    _button = new QAction("&DotAnnotation", this);
    _button->setObjectName(QString::fromStdString(name()));
    if (QApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark) {
        _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/dot_dark.png")));
    }
    else {
        _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/dot.png")));
    }
    _button->setShortcut(QKeySequence("d"));
  }
  return _button;
}

std::string DotAnnotationTool::name() {
  return std::string("dotannotation");
}