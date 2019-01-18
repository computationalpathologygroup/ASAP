#include "RectangleAnnotationTool.h"
#include "QtAnnotation.h"
#include "AnnotationWorkstationExtensionPlugin.h"
#include <QAction>
#include <QPen>
#include <QGraphicsItem>
#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QFormLayout>
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QApplication>
#include "PolyQtAnnotation.h"
#include "annotation/Annotation.h"
#include "../PathologyViewer.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include <cmath>

RectangleAnnotationTool::QGraphicsTextItemWithBackground::QGraphicsTextItemWithBackground(const QString &text) :
QGraphicsTextItem(text) { }

QRectF RectangleAnnotationTool::QGraphicsTextItemWithBackground::boundingRect() const {
  return QGraphicsTextItem::boundingRect().adjusted(-5, -5, 5, 5);
}

void RectangleAnnotationTool::QGraphicsTextItemWithBackground::paint(QPainter *painter, const QStyleOptionGraphicsItem *o, QWidget *w) {
  painter->setPen(Qt::NoPen);
  painter->setBrush(QBrush(QColor(0, 0, 0, 75)));
  painter->drawRect(boundingRect().adjusted(-5, -5, 5, 5));
  painter->setBrush(QBrush(Qt::white));
  QGraphicsTextItem::paint(painter, o, w);
}

RectangleAnnotationTool::RectangleAnnotationTool(AnnotationWorkstationExtensionPlugin* annotationPlugin, PathologyViewer* viewer) :
AnnotationTool(annotationPlugin, viewer),
_activeRect(NULL),
_sizeText(NULL),
_lastAction("")
{
}

void RectangleAnnotationTool::mouseMoveEvent(QMouseEvent *event) {
  if (_viewer) {
    if (_generating) {
      if (!_activeRect) {
        _activeRect = new QGraphicsRectItem();
        _activeRect->setZValue(std::numeric_limits<float>::max());
        _viewer->scene()->addItem(_activeRect);
      }
      if (!_sizeText) {
        _sizeText = new QGraphicsTextItemWithBackground("");
        _sizeText->setZValue(std::numeric_limits<float>::max());
        _sizeText->setFlag(QGraphicsItem::ItemIgnoresTransformations);
        _sizeText->setDefaultTextColor(Qt::white);
        _viewer->scene()->addItem(_sizeText);
      }
      QPointF scenePos = _viewer->mapToScene(event->pos());
      float activeRectWidth = 2 * (scenePos.x() - _start.getX());
      float activeRectHeight = 2 * (scenePos.y() - _start.getY());
      if (std::shared_ptr<MultiResolutionImage> local_img = _annotationPlugin->getCurrentImage().lock()) {
        std::vector<double> spacing = local_img->getSpacing();
        float sizeInPixels = (activeRectWidth / _viewer->getSceneScale()) * (activeRectHeight / _viewer->getSceneScale());
        if (spacing.size() > 0) {
          float sizeInUnits = sizeInPixels * (spacing[0] * spacing[0]);
          QString unit = " um<sup>2</sup>";
          if (sizeInUnits > 1000000) {
            sizeInUnits /= 1000000;
            unit = " mm<sup>2</sup>";
          }
          _sizeText->setHtml(QString::number(sizeInUnits, 'g', 4) + unit);
        }
        else {
          _sizeText->setHtml(QString::number(sizeInPixels) + QString(" pixels"));
        }
        _sizeText->setPos(scenePos.x() + 20 / _viewer->transform().m11(), scenePos.y() + 20 / _viewer->transform().m11());
      }
      _activeRect->setPen(QPen(QBrush(Qt::green), 4. / _viewer->transform().m11()));
      _activeRect->setRect(_start.getX() - activeRectWidth / 2., _start.getY() - activeRectHeight / 2., activeRectWidth, activeRectHeight);
    }
    AnnotationTool::mouseMoveEvent(event);
    event->accept();
  }
}

void RectangleAnnotationTool::keyPressEvent(QKeyEvent *event) {
  AnnotationTool::keyPressEvent(event);
  if (event->key() == Qt::Key::Key_Delete) {
    if (!_generating && _annotationPlugin->getActiveAnnotation()) {
      if (_annotationPlugin->getActiveAnnotation()->getAnnotation()->getCoordinates().size() <= 2) {
        _annotationPlugin->deleteAnnotation(_annotationPlugin->getActiveAnnotation());
      }
    }
  }
}

void RectangleAnnotationTool::cancelAnnotation() {
  if (_generating) {
    AnnotationTool::cancelAnnotation();
    if (_activeRect) {
      _activeRect->hide();
      _viewer->scene()->removeItem(_activeRect);
      delete _activeRect;
      _activeRect = NULL;
    }
    if (_sizeText) {
      _sizeText->hide();
      _viewer->scene()->removeItem(_sizeText);
      delete _sizeText;
      _sizeText = NULL;
    }
  }
}

void RectangleAnnotationTool::mouseDoubleClickEvent(QMouseEvent *event) {
  if (std::shared_ptr<MultiResolutionImage> local_img = _annotationPlugin->getCurrentImage().lock()) {
    std::vector<double> spacing = local_img->getSpacing();
    QString suffix(" pixels");
    if (spacing.size() > 1) {
      suffix = QString::fromUtf8(u8" \u03bcm");
    }
    QDialog* createRectDiaglog = new QDialog();
    createRectDiaglog->setWindowTitle("Specify width and height for rectangular annotations");
    QVBoxLayout* dialogLayout = new QVBoxLayout();
    QFormLayout* heightWidthLayout = new QFormLayout();
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    QDoubleSpinBox* widthSpinBox = new QDoubleSpinBox();
    widthSpinBox->setMinimum(1);
    widthSpinBox->setMaximum(100000);
    widthSpinBox->setValue(1000);
    widthSpinBox->setSingleStep(100);
    widthSpinBox->setSuffix(suffix);
    widthSpinBox->setObjectName("Width");
    heightWidthLayout->addRow("Width", widthSpinBox);
    QDoubleSpinBox* heightSpinBox = new QDoubleSpinBox();
    heightSpinBox->setMinimum(1);
    heightSpinBox->setMaximum(100000);
    heightSpinBox->setValue(1000);
    heightSpinBox->setSingleStep(100);
    heightSpinBox->setSuffix(suffix);
    heightSpinBox->setObjectName("Height");
    heightWidthLayout->addRow("Height", heightSpinBox);
    dialogLayout->addLayout(heightWidthLayout);
    QPushButton* cancel = new QPushButton("Cancel");
    QPushButton* ok = new QPushButton("Ok");
    cancel->setDefault(true);
    connect(cancel, SIGNAL(clicked()), createRectDiaglog, SLOT(reject()));
    connect(ok, SIGNAL(clicked()), createRectDiaglog, SLOT(accept()));
    buttonLayout->addWidget(cancel);
    buttonLayout->addWidget(ok);
    dialogLayout->addLayout(buttonLayout);
    createRectDiaglog->setLayout(dialogLayout);
    int rval = createRectDiaglog->exec();
    if (rval == 1) {
      float rectWidth = widthSpinBox->value();
      float rectHeight = heightSpinBox->value();
      if (spacing.size() > 1) {
        rectWidth /= spacing[0];
        rectHeight /= spacing[1];
      }
      QPointF scenePos = _viewer->mapToScene(event->pos());
      this->addCoordinate(QPointF(scenePos.x() + (_viewer->getSceneScale() * rectWidth / 2), scenePos.y() + (_viewer->getSceneScale() * rectHeight / 2)));
    }
    else {
      this->cancelAnnotation();
    }
  }
}

void RectangleAnnotationTool::addCoordinate(const QPointF& scenePos) {
  if (_annotationPlugin->getGeneratedAnnotation()->getAnnotation()->getCoordinates().size() > 0) {
    _annotationPlugin->getGeneratedAnnotation()->getAnnotation()->clearCoordinates();
    float width = std::abs(2 * (scenePos.x() - _start.getX()) / _viewer->getSceneScale());
    float height = std::abs(2 * (scenePos.y() - _start.getY()) / _viewer->getSceneScale());
    float left = (_start.getX() / _viewer->getSceneScale()) - width / 2.;
    float top = (_start.getY() / _viewer->getSceneScale()) - height / 2.;    
    _annotationPlugin->getGeneratedAnnotation()->getAnnotation()->addCoordinate(left, top);
    _annotationPlugin->getGeneratedAnnotation()->getAnnotation()->addCoordinate(left + width, top);
    _annotationPlugin->getGeneratedAnnotation()->getAnnotation()->addCoordinate(left + width, top + height);
    _annotationPlugin->getGeneratedAnnotation()->getAnnotation()->addCoordinate(left, top + height);
    _annotationPlugin->getGeneratedAnnotation()->setPos(QPointF(left * _viewer->getSceneScale(), top * _viewer->getSceneScale()));
    _annotationPlugin->finishAnnotation();
    if (_activeRect) {
      _viewer->scene()->removeItem(_activeRect);
      delete _activeRect;
      _activeRect = NULL;
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

QAction* RectangleAnnotationTool::getToolButton() {
  if (!_button) {
    _button = new QAction("RectangleAnnotation", this);
    _button->setObjectName(QString::fromStdString(name()));
    _button->setIcon(QIcon(QPixmap(":/AnnotationWorkstationExtensionPlugin_icons/rectangle.png")));
  }
  return _button;
}

std::string RectangleAnnotationTool::name() {
  return std::string("rectangleannotation");
}
