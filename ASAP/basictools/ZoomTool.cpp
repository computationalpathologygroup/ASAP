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
  PathologyViewer* viewer(_controller->GetMasterViewer());
  if (viewer && _zooming) {
    float delta = event->pos().y() - _prevZoomPoint.y();
    _prevZoomPoint = event->pos();
    _accumZoom += delta/20.;
    if (_accumZoom * delta < 0) {
      _accumZoom = 0;
    }
    if (_accumZoom > 1.0 || _accumZoom < -1.0) {
		_controller->Zoom(_accumZoom);
      _accumZoom = 0;
    }
    event->accept();
  }
}

void ZoomTool::mousePressEvent(QMouseEvent *event) {
  PathologyViewer* viewer(_controller->GetMasterViewer());
  if (viewer) {
    _zooming = true;
    _accumZoom = 0;
    _prevZoomPoint = event->pos();
	viewer->setCursor(QCursor(Qt::CursorShape::SizeVerCursor));
	_controller->Zoom(0, event->pos(), viewer->mapToScene(event->pos()));
    event->accept();
  }
}

void ZoomTool::mouseReleaseEvent(QMouseEvent *event) {
  PathologyViewer* viewer(_controller->GetMasterViewer());
  if (viewer) {
    _zooming = false;
    _prevZoomPoint = QPoint(0, 0);
	viewer->setCursor(Qt::ArrowCursor);
    event->accept();
  }
}

QAction* ZoomTool::getToolButton() {
  if (!_button) {
    _button = new QAction("Zoom", this);
    _button->setObjectName(QString::fromStdString(name()));
    _button->setIcon(QIcon(QPixmap(":/basictools_icons/zoom.png")));
  }
  return _button;
}

std::string ZoomTool::name() {
  return std::string("zoom");
}