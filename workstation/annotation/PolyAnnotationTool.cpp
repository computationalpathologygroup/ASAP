#include "PolyAnnotationTool.h"
#include "QtAnnotation.h"
#include "AnnotationWorkstationExtensionPlugin.h"
#include <QAction>
#include <QPen>
#include <QGraphicsLineItem>
#include <QGraphicsItem>
#include "PolyQtAnnotation.h"
#include "Annotation.h"
#include "../PathologyViewer.h"
#include <math.h>
#include <numeric>
#include <iostream>

PolyAnnotationTool::PolyAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) : 
  ToolPluginInterface(), 
  _annotationPlugin(annotationPlugin),
  _start(-1,-1),
  _last(-1, -1),
  _generating(false),
  _activeLine(NULL),
  _startSelectionMove(false),
  _moveStart(-1,-1)
{
  _viewer = viewer;
}

void PolyAnnotationTool::mouseMoveEvent(QMouseEvent *event) {
  if (_viewer) {
    if (_generating) {
      if (!_activeLine) {
        _activeLine = new QGraphicsLineItem();
        _activeLine->setZValue(std::numeric_limits<float>::max());
        _viewer->scene()->addItem(_activeLine);
      }
      QPointF scenePos = _viewer->mapToScene(event->pos());
      _activeLine->setPen(QPen(QBrush(Qt::green), 3. / _viewer->transform().m11(), Qt::PenStyle::DashLine));
      _activeLine->setLine(_last.getX(), _last.getY(), scenePos.x(), scenePos.y());
    }
    else if (_startSelectionMove && event->modifiers() == Qt::AltModifier) {
      QPointF scenePos = _viewer->mapToScene(event->pos());
      QList<QtAnnotation*> selected = _annotationPlugin->getSelectedAnnotations();
      for (QList<QtAnnotation*>::iterator it = selected.begin(); it != selected.end(); ++it) {
        QPointF delta = (scenePos - _moveStart);
        (*it)->moveCoordinatesBy(Point(delta.x(), delta.y()));
      }
      _moveStart = scenePos;
    }
    else if (_startSelectionMove) {
      QPointF scenePos = _viewer->mapToScene(event->pos());
      PolyQtAnnotation* active = dynamic_cast<PolyQtAnnotation*>(_annotationPlugin->getActiveAnnotation());
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

void PolyAnnotationTool::mouseDoubleClickEvent(QMouseEvent *event) {
  QPointF scenePos = _viewer->mapToScene(event->pos());
  if (!_generating) {
    PolyQtAnnotation* selected = dynamic_cast<PolyQtAnnotation*>(this->_viewer->itemAt(event->pos()));
    if (selected) {
      PolyQtAnnotation* active = dynamic_cast<PolyQtAnnotation*>(_annotationPlugin->getActiveAnnotation());
      if (active && active->getEditable()) {
        if (active == selected) {
          std::vector<Point> coords = active->getAnnotation()->getCoordinates();
          bool hitSeedPoint = false;
          for (std::vector<Point>::const_iterator it = coords.begin(); it != coords.end(); ++it) {
            if (pow(it->getX()*_viewer->getSceneScale() - scenePos.x(), 2) + pow(it->getY()*_viewer->getSceneScale() - scenePos.y(), 2) < (36. / _viewer->transform().m11())) {
              hitSeedPoint = true;
              break;
            }
          }
          if (!hitSeedPoint) {
            // Figure out between which seed points a point should be added
            QPointF imagePos = scenePos / _viewer->getSceneScale();
            std::pair<int, int> indices = active->seedPointsContaininPathPoint(scenePos);
            if (indices.first >= 0) {
              active->insertCoordinate(indices.second, Point(imagePos.x(), imagePos.y()));
            }
          }
        }
      }
    }
  }
}

void PolyAnnotationTool::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key::Key_Escape) {
    cancelAnnotation();
  }
  else if (event->key() == Qt::Key::Key_Delete && event->modifiers() == Qt::ShiftModifier) {
    QList<QtAnnotation*> selectedAnnotations = _annotationPlugin->getSelectedAnnotations();
    _annotationPlugin->clearSelection();
    for (QList<QtAnnotation*>::iterator it = selectedAnnotations.begin(); it != selectedAnnotations.end(); ++it) {
      _annotationPlugin->deleteAnnotation(*it);
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
          if (_activeLine) {
            _activeLine->setLine(_last.getX(), _last.getY(), _activeLine->line().p2().x(), _activeLine->line().p2().y());
          }
        }
      }
    }
    else if (_annotationPlugin->getActiveAnnotation() && _annotationPlugin->getActiveAnnotation()->getActiveSeedPoint() > -1) {
      int activeSeedPoint = _annotationPlugin->getActiveAnnotation()->getActiveSeedPoint();
      _annotationPlugin->getActiveAnnotation()->removeCoordinate(activeSeedPoint);
      if (activeSeedPoint - 1 >= 0) {
        _annotationPlugin->getActiveAnnotation()->setActiveSeedPoint(activeSeedPoint);
      }
    }
    else if (_annotationPlugin->getActiveAnnotation()) {
      _annotationPlugin->getActiveAnnotation()->removeCoordinate(-1);
    }
  }
}

void PolyAnnotationTool::cancelAnnotation() {
  if (_generating) {
    _annotationPlugin->finishAnnotation(true);
    if (_activeLine) {
      _activeLine->hide();
      _viewer->scene()->removeItem(_activeLine);
      delete _activeLine;
      _activeLine = NULL;
    }
    _start = Point(-1, -1);
    _last = _start;
    _generating = false;
  }
}

void PolyAnnotationTool::mousePressEvent(QMouseEvent *event) {
  if (_viewer) {
    QPointF scenePos = _viewer->mapToScene(event->pos());
    if (!_generating) {
      PolyQtAnnotation* selected = dynamic_cast<PolyQtAnnotation*>(this->_viewer->itemAt(event->pos()));
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
          PolyQtAnnotation* active = dynamic_cast<PolyQtAnnotation*>(_annotationPlugin->getActiveAnnotation());
          if (active) {
            if (active == selected) {
              std::vector<Point> coords = active->getAnnotation()->getCoordinates();
              bool hitSeedPoint = false;
              for (std::vector<Point>::const_iterator it = coords.begin(); it != coords.end(); ++it) {
                if (pow(it->getX()*_viewer->getSceneScale() - scenePos.x(), 2) + pow(it->getY()*_viewer->getSceneScale() - scenePos.y(), 2) < (36. / _viewer->transform().m11())) {
                  active->setActiveSeedPoint(it - coords.begin());
                  _startSelectionMove = true;
                  _moveStart = scenePos;
                  hitSeedPoint = true;
                  break;
                }
              }
              if (!hitSeedPoint) {
                active->clearActiveSeedPoint();
              }
            }
          }
        }
        return;
      }
      else {
        _annotationPlugin->clearSelection();
        _annotationPlugin->startAnnotation(scenePos.x(), scenePos.y(), name());
        _generating = true;
        _start = Point(scenePos.x(), scenePos.y());
        _last = _start;
      }
    }
    else if (_generating) {
      if (_annotationPlugin->getGeneratedAnnotation()->getAnnotation()->getCoordinates().size() > 2 && pow(scenePos.x() - _start.getX(), 2) + pow(scenePos.y() - _start.getY(), 2) < (36. / _viewer->transform().m11())) {
        _annotationPlugin->finishAnnotation();
        if (_activeLine) {
          _viewer->scene()->removeItem(_activeLine);
          delete _activeLine;
          _activeLine = NULL;
        }
        _start = Point(-1,-1);
        _last = _start;
        _generating = false;
      }
      else {
        _annotationPlugin->getGeneratedAnnotation()->addCoordinate(scenePos.x() / _viewer->getSceneScale(), scenePos.y() / _viewer->getSceneScale());
        _last = Point(scenePos.x(), scenePos.y());
      }
    }
    event->accept();
  }
}

void PolyAnnotationTool::mouseReleaseEvent(QMouseEvent *event) {
  if (_viewer) {
    if (_startSelectionMove) {
      _startSelectionMove = false;
      _moveStart = QPointF(-1, -1);
    }
    event->accept();
  }
}

QAction* PolyAnnotationTool::getToolButton() {
  if (!_button) {
    _button = new QAction("PolyAnnotation", this);
    _button->setObjectName(QString::fromStdString(name()));
    _button->setIcon(QIcon(QPixmap(":/poly.png")));
  }
  return _button;
}

std::string PolyAnnotationTool::name() {
  return std::string("polyannotation");
}