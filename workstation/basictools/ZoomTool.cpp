#include "ZoomTool.h"
#include <QAction>
#include "../PathologyViewer.h"
#include <iostream>

ZoomTool::ZoomTool() :
_zooming(false),
_accumZoom(0),
_prevZoomPoint(QPoint(0, 0))
{
}

void ZoomTool::mouseMoveEvent(QMouseEvent *event) {
  if (_viewer && _zooming) {
    float delta = event->pos().y() - _prevZoomPoint.y();
    _prevZoomPoint = event->pos();
    _accumZoom += delta/20.;
    if (_accumZoom * delta < 0) {
      _accumZoom = 0;
    }
    if (_accumZoom > 1.0 || _accumZoom < -1.0) {
      _viewer->zoom(_accumZoom);
      _accumZoom = 0;
    }
    event->accept();
  }
}

void ZoomTool::mousePressEvent(QMouseEvent *event) {
  if (_viewer) {
    _zooming = true;
    _accumZoom = 0;
    _prevZoomPoint = event->pos();
    _viewer->setCursor(QCursor(Qt::CursorShape::SizeVerCursor));
    event->accept();
  }
}

void ZoomTool::mouseReleaseEvent(QMouseEvent *event) {
  if (_viewer) {
    _zooming = false;
    _prevZoomPoint = QPoint(0, 0);
    _viewer->setCursor(Qt::ArrowCursor);
    event->accept();
  }
}

QAction* ZoomTool::getToolButton() {
  if (!_button) {
    _button = new QAction("Zoom", this);
    _button->setObjectName(QString::fromStdString(name()));
    _button->setIcon(QIcon(QPixmap(":/zoom.png")));
  }
  return _button;
}

std::string ZoomTool::name() {
  return std::string("zoom");
}