#include "PointSetAnnotationTool.h"
#include "QtAnnotation.h"
#include "AnnotationWorkstationExtensionPlugin.h"
#include <QAction>
#include <QPen>
#include <QGraphicsLineItem>
#include <QGraphicsItem>
#include "PointSetQtAnnotation.h"
#include "Annotation.h"
#include "../PathologyViewer.h"
#include <math.h>
#include <numeric>
#include <iostream>
#include <QTimeLine>

PointSetAnnotationTool::PointSetAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) :
ToolPluginInterface(),
_annotationPlugin(annotationPlugin),
_start(-1, -1),
_last(-1, -1),
_generating(false),
_startSelectionMove(false),
_moveStart(-1, -1)
{
  _viewer = viewer;
}

void PointSetAnnotationTool::mouseMoveEvent(QMouseEvent *event) {
  if (_viewer) {
    if (_generating) {
      QPointF scenePos = _viewer->mapToScene(event->pos());
      if (event->buttons() == Qt::LeftButton) {
        if (QLineF(_viewer->mapFromScene(scenePos), _viewer->mapFromScene(QPointF(_last.getX(), _last.getY()))).length() > 40) {
          addCoordinate(scenePos);
        }
      }
    }
    else if (_startSelectionMove && event->modifiers() == Qt::AltModifier) {
      QPointF scenePos = _viewer->mapToScene(event->pos());
      QSet<QtAnnotation*> selected = _annotationPlugin->getSelectedAnnotations();
      for (QSet<QtAnnotation*>::iterator it = selected.begin(); it != selected.end(); ++it) {
        QPointF delta = (scenePos - _moveStart);
        (*it)->moveCoordinatesBy(Point(delta.x(), delta.y()));
      }
      _moveStart = scenePos;
    }
    else if (_startSelectionMove) {
      QPointF scenePos = _viewer->mapToScene(event->pos());
      PointSetQtAnnotation* active = dynamic_cast<PointSetQtAnnotation*>(_annotationPlugin->getActiveAnnotation());
      if (active && active->getEditable()) {
        int activeSeedPoint = active->getActiveSeedPoint();
        if (activeSeedPoint >= 0) {
          QPointF delta = (scenePos - _moveStart);
          active->moveCoordinateBy(activeSeedPoint, Point(delta.x(), delta.y()));
          _moveStart = scenePos;
        }
      }
    }
    event->accept();
  }
}

void PointSetAnnotationTool::mouseDoubleClickEvent(QMouseEvent *event) {
  if (_generating) {
    _annotationPlugin->finishAnnotation();
    _start = Point(-1, -1);
    _last = _start;
    _generating = false;
  }
}

void PointSetAnnotationTool::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key::Key_Escape) {
    cancelAnnotation();
  }
  else if (event->key() == Qt::Key::Key_Delete && event->modifiers() == Qt::ShiftModifier) {
    if (_generating) {
      cancelAnnotation();
    }
    else {
      QSet<QtAnnotation*> selectedAnnotations = _annotationPlugin->getSelectedAnnotations();
      for (QSet<QtAnnotation*>::iterator it = selectedAnnotations.begin(); it != selectedAnnotations.end(); ++it) {
        _annotationPlugin->deleteAnnotation(*it);
      }
    }
  }
  else if (event->key() == Qt::Key::Key_Delete) {
    if (_generating) {
      if (_annotationPlugin->getGeneratedAnnotation()->getAnnotation()->getCoordinates().size() < 2) {
        cancelAnnotation();
      }
      else {
        _annotationPlugin->getGeneratedAnnotation()->removeCoordinate(-1);
        if (!_annotationPlugin->getGeneratedAnnotation()->getAnnotation()->getCoordinates().empty()) {
          Point prev = _annotationPlugin->getGeneratedAnnotation()->getAnnotation()->getCoordinate(-1);
          _last = Point(prev.getX() * _viewer->getSceneScale(), prev.getY() * _viewer->getSceneScale());
        }
      }
    }
    else if (_annotationPlugin->getActiveAnnotation()) {
      if (_annotationPlugin->getActiveAnnotation()->getAnnotation()->getCoordinates().size() <= 1) {
        _annotationPlugin->deleteAnnotation(_annotationPlugin->getActiveAnnotation());
      }
      else if (_annotationPlugin->getActiveAnnotation()->getActiveSeedPoint() > -1) {
        int activeSeedPoint = _annotationPlugin->getActiveAnnotation()->getActiveSeedPoint();
        _annotationPlugin->getActiveAnnotation()->removeCoordinate(activeSeedPoint);
        if (activeSeedPoint - 1 >= 0) {
          _annotationPlugin->getActiveAnnotation()->setActiveSeedPoint(activeSeedPoint - 1);
        }
        else {
          _annotationPlugin->getActiveAnnotation()->setActiveSeedPoint(_annotationPlugin->getActiveAnnotation()->getAnnotation()->getCoordinates().size() - 1);
        }
      }
      else if (_annotationPlugin->getActiveAnnotation()) {
        _annotationPlugin->getActiveAnnotation()->removeCoordinate(-1);
      }
    }
  }
  else if (event->key() == Qt::Key::Key_Z) {
    if (_annotationPlugin->getActiveAnnotation()) {
      QTimeLine * anim = new QTimeLine(500);

      _start_zoom = _viewer->mapToScene(_viewer->viewport()->rect()).boundingRect();
      _end_zoom = _annotationPlugin->getActiveAnnotation()->mapToScene(_annotationPlugin->getActiveAnnotation()->boundingRect()).boundingRect();
      anim->setFrameRange(0, 100);
      anim->setUpdateInterval(5);

      connect(anim, SIGNAL(valueChanged(qreal)), SLOT(zoomToAnnotation(qreal)));
      connect(anim, SIGNAL(finished()), SLOT(zoomToAnnotationFinished()));
      anim->start();
    }
  }
}

void PointSetAnnotationTool::setActive(bool active) {
  if (!active) {
    if (_generating) {
      this->cancelAnnotation();
    }
  }
  _active = active;
}

void PointSetAnnotationTool::zoomToAnnotation(qreal val) {
  QRectF current = QRectF(_start_zoom.topLeft() + val*(_end_zoom.topLeft() - _start_zoom.topLeft()), _start_zoom.bottomRight() + val*(_end_zoom.bottomRight() - _start_zoom.bottomRight()));
  _viewer->fitInView(current, Qt::AspectRatioMode::KeepAspectRatio);
}

void PointSetAnnotationTool::zoomToAnnotationFinished() {
  sender()->~QObject();
}

void PointSetAnnotationTool::cancelAnnotation() {
  if (_generating) {
    _annotationPlugin->finishAnnotation(true);
    _start = Point(-1, -1);
    _last = _start;
    _generating = false;
  }
}

void PointSetAnnotationTool::mousePressEvent(QMouseEvent *event) {
  if (_viewer) {
    QPointF scenePos = _viewer->mapToScene(event->pos());
    if (!_generating) {
      PointSetQtAnnotation* selected = dynamic_cast<PointSetQtAnnotation*>(this->_viewer->itemAt(event->pos()));
      if (selected) {
        if (event->modifiers() != Qt::ControlModifier && event->modifiers() != Qt::AltModifier) {
          _annotationPlugin->clearSelection();
          if (!selected->isSelected()) {
            _annotationPlugin->addAnnotationToSelection(selected);
          }
          else {
            _annotationPlugin->removeAnnotationFromSelection(selected);
          }
        }
        else if (event->modifiers() != Qt::AltModifier) {
          if (!selected->isSelected()) {
            _annotationPlugin->addAnnotationToSelection(selected);
          }
          else {
            _annotationPlugin->removeAnnotationFromSelection(selected);
          }
        }
        if (event->modifiers() == Qt::AltModifier) {
          _startSelectionMove = true;
          _moveStart = scenePos;
        }
        else {
          PointSetQtAnnotation* active = dynamic_cast<PointSetQtAnnotation*>(_annotationPlugin->getActiveAnnotation());
          if (active) {
            if (active == selected) {
              int lastIndexPoint = active->getLastClickedPointIndex();
              if (lastIndexPoint >= 0) {
                active->setActiveSeedPoint(lastIndexPoint);
                _startSelectionMove = true;
                _moveStart = scenePos;
              }
              if (lastIndexPoint < 0) {
                active->clearActiveSeedPoint();
              }
            }
          }
        }
        return;
      }
      else {
        _annotationPlugin->startAnnotation(scenePos.x(), scenePos.y(), name());
        _generating = true;
        _start = Point(scenePos.x(), scenePos.y());
        _last = _start;
      }
    }
    else if (_generating) {
      addCoordinate(scenePos);
    }
    event->accept();
  }
}

void PointSetAnnotationTool::addCoordinate(const QPointF& scenePos) {
  if (_annotationPlugin->getGeneratedAnnotation()->getAnnotation()->getCoordinates().size() > 2 && QLineF(_viewer->mapFromScene(QPointF(_start.getX(), _start.getY())), _viewer->mapFromScene(scenePos)).length() < 12) {
    _annotationPlugin->finishAnnotation();
    _start = Point(-1, -1);
    _last = _start;
    _generating = false;
  }
  else {
    _annotationPlugin->getGeneratedAnnotation()->addCoordinate(scenePos.x() / _viewer->getSceneScale(), scenePos.y() / _viewer->getSceneScale());
    _last = Point(scenePos.x(), scenePos.y());
  }
}

void PointSetAnnotationTool::mouseReleaseEvent(QMouseEvent *event) {
  if (_viewer) {
    if (_startSelectionMove) {
      _startSelectionMove = false;
      _moveStart = QPointF(-1, -1);
    }
    event->accept();
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