#include "DotAnnotationTool.h"
#include "DotQtAnnotation.h"
#include <QAction>
#include "../PathologyViewer.h"
#include "AnnotationWorkstationExtensionPlugin.h"
#include "core/Point.h"

DotAnnotationTool::DotAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) : 
  ToolPluginInterface(),
  _annotationPlugin(annotationPlugin),
  _startSelectionMove(false),
  _moveStart(-1, -1)
{
  _viewer = viewer;
}

void DotAnnotationTool::mouseMoveEvent(QMouseEvent *event) {
  if (_viewer) {
    if (_startSelectionMove && event->modifiers() == Qt::AltModifier) {
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
      DotQtAnnotation* active = dynamic_cast<DotQtAnnotation*>(_annotationPlugin->getActiveAnnotation());
      if (active && active->getEditable()) {
        QPointF delta = (scenePos - _moveStart);
        active->moveCoordinateBy(Point(delta.x(), delta.y()));
        _moveStart = scenePos;
      }
    }
    event->accept();
  }
}

void DotAnnotationTool::mousePressEvent(QMouseEvent *event) {
  if (_viewer) {
    QPointF scenePos = _viewer->mapToScene(event->pos());
    DotQtAnnotation* selected = dynamic_cast<DotQtAnnotation*>(this->_viewer->itemAt(event->pos()));
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
      selected->setActiveSeedPoint(0);
      _startSelectionMove = true;
      _moveStart = scenePos;
      return;
    }
    else {
      QPointF scenePos = _viewer->mapToScene(event->pos());
      _annotationPlugin->startAnnotation(scenePos.x(), scenePos.y(), name());
      _annotationPlugin->finishAnnotation();
    }
  }
  event->accept();
}

void DotAnnotationTool::mouseReleaseEvent(QMouseEvent *event) {
  if (_viewer) {
    if (_startSelectionMove) {
      _startSelectionMove = false;
      _moveStart = QPointF(-1, -1);
    }
    event->accept();
  }
}

void DotAnnotationTool::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key::Key_Delete) {
    QSet<QtAnnotation*> selectedAnnotations = _annotationPlugin->getSelectedAnnotations();
    for (QSet<QtAnnotation*>::iterator it = selectedAnnotations.begin(); it != selectedAnnotations.end(); ++it) {
      _annotationPlugin->deleteAnnotation(*it);
    }
  }
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