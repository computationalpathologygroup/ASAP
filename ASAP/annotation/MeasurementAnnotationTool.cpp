#include "MeasurementAnnotationTool.h"
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
#include "multiresolutionimageinterface/MultiResolutionImage.h"

MeasurementAnnotationTool::QGraphicsTextItemWithBackground::QGraphicsTextItemWithBackground(const QString &text) :
QGraphicsTextItem(text) { }

QRectF MeasurementAnnotationTool::QGraphicsTextItemWithBackground::boundingRect() const {
  return QGraphicsTextItem::boundingRect().adjusted(-5, -5, 5, 5);
}

void MeasurementAnnotationTool::QGraphicsTextItemWithBackground::paint(QPainter *painter, const QStyleOptionGraphicsItem *o, QWidget *w) {
  painter->setPen(Qt::NoPen);
  painter->setBrush(QBrush(QColor(0, 0, 0, 75)));
  painter->drawRect(boundingRect().adjusted(-5, -5, 5, 5));
  painter->setBrush(QBrush(Qt::white));
  QGraphicsTextItem::paint(painter, o, w);
}

MeasurementAnnotationTool::MeasurementAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) :
AnnotationTool(annotationPlugin, viewer),
_activeLine(NULL),
_sizeText(NULL)
{
}

void MeasurementAnnotationTool::mouseMoveEvent(QMouseEvent *event) {
  if (_viewer) {
    if (_generating) {
      if (!_activeLine) {
        _activeLine = new QGraphicsLineItem();
        _activeLine->setZValue(std::numeric_limits<float>::max());
        _viewer->scene()->addItem(_activeLine);
      }
      if (!_sizeText) {
        _sizeText = new QGraphicsTextItemWithBackground("");
        _sizeText->setZValue(std::numeric_limits<float>::max());
        _sizeText->setFlag(QGraphicsItem::ItemIgnoresTransformations);
        _sizeText->setDefaultTextColor(Qt::white);
        _viewer->scene()->addItem(_sizeText);
      }
      QPointF scenePos = _viewer->mapToScene(event->pos());
      if (std::shared_ptr<MultiResolutionImage> local_img = _annotationPlugin->getCurrentImage().lock()) {
        std::vector<double> spacing = local_img->getSpacing();
        float sizeInPixels = sqrt(pow(scenePos.x() / _viewer->getSceneScale() - _last.getX() / _viewer->getSceneScale(), 2) + pow(scenePos.y() / _viewer->getSceneScale() - _last.getY() / _viewer->getSceneScale(), 2));
        if (spacing.size() > 0) {
          float sizeInUnits = sizeInPixels * spacing[0];
          QString unit = " um";
          if (sizeInUnits > 1000) {
            sizeInUnits /= 1000;
            unit = " mm";
          }
          _sizeText->setPlainText(QString::number(sizeInUnits, 'g', 4) + unit);
        }
        else {
          _sizeText->setPlainText(QString::number(sizeInPixels) + QString(" pixels"));
        }
        _sizeText->setPos(scenePos.x() + 20 / _viewer->transform().m11(), scenePos.y() + 20 / _viewer->transform().m11());
      }
      _activeLine->setPen(QPen(QBrush(Qt::black), 3. / _viewer->transform().m11()));
      _activeLine->setLine(_last.getX(), _last.getY(), scenePos.x(), scenePos.y());
    }
    AnnotationTool::mouseMoveEvent(event);
    event->accept();
  }
}

void MeasurementAnnotationTool::keyPressEvent(QKeyEvent *event) {
  AnnotationTool::keyPressEvent(event);
  if (event->key() == Qt::Key::Key_Delete) {
    if (!_generating && _annotationPlugin->getActiveAnnotation()) {
      if (_annotationPlugin->getActiveAnnotation()->getAnnotation()->getCoordinates().size() <= 2) {
        _annotationPlugin->deleteAnnotation(_annotationPlugin->getActiveAnnotation());
      }
    }
  }
}

void MeasurementAnnotationTool::cancelAnnotation() {
  if (_generating) {
    AnnotationTool::cancelAnnotation();
    if (_activeLine) {
      _activeLine->hide();
      _viewer->scene()->removeItem(_activeLine);
      delete _activeLine;
      _activeLine = NULL;
    }
    if (_sizeText) {
      _sizeText->hide();
      _viewer->scene()->removeItem(_sizeText);
      delete _sizeText;
      _sizeText = NULL;
    }
  }
}

void MeasurementAnnotationTool::addCoordinate(const QPointF& scenePos) {
  if (_annotationPlugin->getGeneratedAnnotation()->getAnnotation()->getCoordinates().size() > 0) {
    _annotationPlugin->getGeneratedAnnotation()->addCoordinate(scenePos.x() / _viewer->getSceneScale(), scenePos.y() / _viewer->getSceneScale());
    _annotationPlugin->finishAnnotation();
    if (_activeLine) {
      _viewer->scene()->removeItem(_activeLine);
      delete _activeLine;
      _activeLine = NULL;
    }
    if (_sizeText) {
      _viewer->scene()->removeItem(_sizeText);
      delete _sizeText;
      _sizeText = NULL;
    }
    _start = Point(-1, -1);
    _last = _start;
    _generating = false;
  }
  else {
    _annotationPlugin->getGeneratedAnnotation()->addCoordinate(scenePos.x() / _viewer->getSceneScale(), scenePos.y() / _viewer->getSceneScale());
    _last = Point(scenePos.x(), scenePos.y());
  }
}

QAction* MeasurementAnnotationTool::getToolButton() {
  if (!_button) {
    _button = new QAction("MeasurementAnnotation", this);
    _button->setObjectName(QString::fromStdString(name()));
    _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/measure.png")));
  }
  return _button;
}

std::string MeasurementAnnotationTool::name() {
  return std::string("measurementannotation");
}