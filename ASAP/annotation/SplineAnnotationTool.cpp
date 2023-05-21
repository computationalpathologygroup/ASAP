#include "SplineAnnotationTool.h"
#include <QAction>
#include <QPixmap>
#include <QIcon>
#include <QApplication>
#include <QStyleHints>
#include "AnnotationWorkstationExtensionPlugin.h"
#include "../PathologyViewer.h"

SplineAnnotationTool::SplineAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) :
PolyAnnotationTool(annotationPlugin, viewer)
{
}

QAction* SplineAnnotationTool::getToolButton() {
  if (!_button) {
    _button = new QAction("&SplineAnnotation", this);
    _button->setObjectName(QString::fromStdString(name()));
    if (QApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark) {
        _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/spline_dark.png")));
    }
    else {
        _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/spline.png")));
    }
    _button->setShortcut(QKeySequence("s"));
  }
  return _button;
}

std::string SplineAnnotationTool::name() {
  return std::string("splineannotation");
}