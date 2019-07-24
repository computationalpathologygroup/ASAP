#include "PolyAnnotationTool.h"
#include "QtAnnotation.h"
#include "AnnotationWorkstationExtensionPlugin.h"
#include <QAction>
#include <QPen>
#include <QGraphicsLineItem>
#include <QGraphicsItem>
#include "PolyQtAnnotation.h"
#include "annotation/Annotation.h"
#include "../PathologyViewer.h"
#include <math.h>
#include <numeric>
#include <iostream>
#include <QTimeLine>

PolyAnnotationTool::PolyAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, ASAP::PathologyViewController& controller) :
  AnnotationTool(annotationPlugin, controller), _activeLine(NULL)
{
}

void PolyAnnotationTool::mouseMoveEvent(QMouseEvent *event) {
  PathologyViewer* viewer(_controller->GetMasterViewer());
  if (viewer) {
    if (_generating) {
      if (!_activeLine) {
        _activeLine = new QGraphicsLineItem();
        _activeLine->setZValue(std::numeric_limits<float>::max());
		viewer->scene()->addItem(_activeLine);
      }
      QPointF scenePos = viewer->mapToScene(event->pos());
      _activeLine->setPen(QPen(QBrush(Qt::green), 3. / viewer->transform().m11(), Qt::PenStyle::DashLine));
      _activeLine->setLine(_last.getX(), _last.getY(), scenePos.x(), scenePos.y());
    }
    AnnotationTool::mouseMoveEvent(event);
    event->accept();
  }
}

void PolyAnnotationTool::mouseDoubleClickEvent(QMouseEvent *event) {
  PathologyViewer* viewer(_controller->GetMasterViewer());
  QPointF scenePos = viewer->mapToScene(event->pos());
  if (!_generating) {
    PolyQtAnnotation* selected = dynamic_cast<PolyQtAnnotation*>(viewer->itemAt(event->pos()));
    if (selected) {
      PolyQtAnnotation* active = dynamic_cast<PolyQtAnnotation*>(_annotationPlugin->getActiveAnnotation());
      if (active && active->getEditable()) {
        if (active == selected) {
          QPointF lineLocation = active->getLastClickedLinePoint();
          if (!lineLocation.isNull()) {
            std::pair<int, int> indices = active->getLastClickedCoordinateIndices();
            if (indices.first >= 0) {
              active->insertCoordinate(indices.second, Point(lineLocation.x(), lineLocation.y()));
            }
          }
        }
      }
    }
  }
}

void PolyAnnotationTool::cancelAnnotation() {
  PathologyViewer* viewer(_controller->GetMasterViewer());
  if (_generating) {
    AnnotationTool::cancelAnnotation();
    if (_activeLine) {
      _activeLine->hide();
	  viewer->scene()->removeItem(_activeLine);
      delete _activeLine;
      _activeLine = NULL;
    }
  }
}

void PolyAnnotationTool::addCoordinate(const QPointF& scenePos) {
  PathologyViewer* viewer(_controller->GetMasterViewer());
  if (_annotationPlugin->getGeneratedAnnotation()->getAnnotation()->getCoordinates().size() > 2 && QLineF(viewer->mapFromScene(QPointF(_start.getX(), _start.getY())), viewer->mapFromScene(scenePos)).length() < 12) {
    _annotationPlugin->finishAnnotation();
    if (_activeLine) {
		viewer->scene()->removeItem(_activeLine);
      delete _activeLine;
      _activeLine = NULL;
    }
    _start = Point(-1, -1);
    _last = _start;
    _generating = false;
  }
  else {
    _annotationPlugin->getGeneratedAnnotation()->addCoordinate(scenePos.x() / viewer->getSceneScale(), scenePos.y() / viewer->getSceneScale());
    _last = Point(scenePos.x(), scenePos.y());
  }
}

QAction* PolyAnnotationTool::getToolButton() {
  if (!_button) {
    _button = new QAction("PolyAnnotation", this);
    _button->setObjectName(QString::fromStdString(name()));
    _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/poly.png")));
  }
  return _button;
}

std::string PolyAnnotationTool::name() {
  return std::string("polyannotation");
}