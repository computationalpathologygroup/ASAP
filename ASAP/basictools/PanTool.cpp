#include "PanTool.h"
#include <QAction>
#include <QApplication>
#include <QStyleHints>
#include "../PathologyViewer.h"

void PanTool::mouseMoveEvent(QMouseEvent *event) {
  if (_viewer) {
    if (_viewer->isPanning()) {
      _viewer->pan(event->pos());
      event->accept();
    }
  }
}

void PanTool::mousePressEvent(QMouseEvent *event) {
  if (_viewer) {
    _viewer->togglePan(true, event->pos());
    event->accept();
  }
}

void PanTool::mouseReleaseEvent(QMouseEvent *event) {
  if (_viewer) {
    _viewer->togglePan(false);
    event->accept();
  }
}

QAction* PanTool::getToolButton() {
  if (!_button) {
    _button = new QAction("Pan", this);
    _button->setObjectName(QString::fromStdString(name()));
    if (QApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark) {
        _button->setIcon(QIcon(QPixmap(":/basictools_icons/pan_dark.png")));
    }
    else {
        _button->setIcon(QIcon(QPixmap(":/basictools_icons/pan.png")));
    }
    _button->setShortcut(QKeySequence("x"));
  }
  return _button;
}

std::string PanTool::name() {
  return std::string("pan");
}