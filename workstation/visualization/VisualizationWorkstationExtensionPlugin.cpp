#include "VisualizationWorkstationExtensionPlugin.h"
#include "../PathologyViewer.h"
#include <QDockWidget>
#include <QtUiTools>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGraphicsPolygonItem>
#include <QPolygonF>
#include "io/multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "core/filetools.h"
#include "annotation/XmlRepository.h"
#include "annotation/Annotation.h"
#include "annotation/AnnotationList.h"

VisualizationWorkstationExtensionPlugin::VisualizationWorkstationExtensionPlugin() :
  WorkstationExtensionPluginInterface(),
  _dockWidget(NULL),
  _likelihoodCheckBox(NULL),
  _foreground(NULL),
  _foregroundScale(1.),
  _opacity(1.0),
  _annotations(NULL),
  _lst(NULL)
{
  _lst = new AnnotationList();
  _annotations = new XmlRepository(_lst);
}

VisualizationWorkstationExtensionPlugin::~VisualizationWorkstationExtensionPlugin() {
  if (_annotations) {
    delete _annotations;
    _annotations = NULL;
  }
  if (_lst) {
    delete _lst;
    _lst = NULL;
  }
  
  if (_foreground) {
    _foregroundScale = 1.;
    emit changeForegroundImage(NULL, _foregroundScale);
    _foreground = NULL;
  }
  _dockWidget = NULL;
}

bool VisualizationWorkstationExtensionPlugin::initialize(PathologyViewer* viewer) {
  _viewer = viewer;
  connect(this, SIGNAL(changeForegroundImage(MultiResolutionImage*, float)), viewer, SLOT(onForegroundImageChanged(MultiResolutionImage*, float)));
  return true;
}

QDockWidget* VisualizationWorkstationExtensionPlugin::getDockWidget() {
  _dockWidget = new QDockWidget("Cancer detection visualization");
  QUiLoader loader;
  QFile file(":/VisualizationWorkstationExtensionPlugin_ui/VisualizationWorkstationExtensionPlugin.ui");
  file.open(QFile::ReadOnly);
  QWidget* content = loader.load(&file, _dockWidget);
  file.close();
  _likelihoodCheckBox = content->findChild<QCheckBox*>("LikelihoodCheckBox");
  QDoubleSpinBox* spinBox = content->findChild<QDoubleSpinBox*>("OpacitySpinBox");
  _segmentationCheckBox = content->findChild<QCheckBox*>("SegmentationCheckBox");
  connect(_likelihoodCheckBox, SIGNAL(toggled(bool)), this, SLOT(onEnableLikelihoodToggled(bool)));
  connect(_segmentationCheckBox, SIGNAL(toggled(bool)), this, SLOT(onEnableSegmentationToggled(bool)));
  connect(spinBox, SIGNAL(valueChanged(double)), this, SLOT(onOpacityChanged(double)));
  _dockWidget->setEnabled(false);
  return _dockWidget;
}

void VisualizationWorkstationExtensionPlugin::onNewImageLoaded(MultiResolutionImage* img, std::string fileName) {
  if (_dockWidget) {
    _dockWidget->setEnabled(true);
  }
  if (!fileName.empty()) {
    std::string base = core::extractBaseName(fileName);
    std::string likImgPth = core::completePath(base + "_likelihood_map.tif", core::extractFilePath(fileName));
    std::string segmXMLPth = core::completePath(base + "_detections.xml", core::extractFilePath(fileName));
    if (core::fileExists(likImgPth)) {
      MultiResolutionImageReader reader;
      if (_foreground) {
        _foregroundScale = 1;
        emit changeForegroundImage(NULL, _foregroundScale);
        _foreground = NULL;
      }
      _foreground = reader.open(likImgPth);
      if (_foreground) {
        std::vector<unsigned long long> dimsBG = img->getDimensions();
        std::vector<unsigned long long> dimsFG = _foreground->getDimensions();
        if (dimsBG[0] / dimsFG[0] == dimsBG[1] / dimsFG[1]) {
          _foregroundScale = dimsBG[0] / dimsFG[0];
          if (_likelihoodCheckBox && _likelihoodCheckBox->isChecked()) {
            emit changeForegroundImage(_foreground, _foregroundScale);
          }
          else {
            emit changeForegroundImage(NULL, _foregroundScale);
          }
          if (_viewer) {
            _viewer->setForegroundOpacity(_opacity);
          }
        }
      }
    }
    if (core::fileExists(segmXMLPth)) {
      _annotations->setSource(segmXMLPth);
      _annotations->load();
      if (_segmentationCheckBox && _segmentationCheckBox->isChecked()) {
        addSegmentationsToViewer();
      }
    }
  }
}

void VisualizationWorkstationExtensionPlugin::onImageClosed() {
  if (!_polygons.empty()) {
    removeSegmentationsFromViewer();
  }
  if (_foreground) {
    _foregroundScale = 1;
    emit changeForegroundImage(NULL, _foregroundScale);
    delete _foreground;
    _foreground = NULL;
  }
  if (_dockWidget) {
    _dockWidget->setEnabled(false);
  }
}

void VisualizationWorkstationExtensionPlugin::onEnableLikelihoodToggled(bool toggled) {
  if (!toggled) {
    emit changeForegroundImage(NULL, _foregroundScale);
  }
  else {
    emit changeForegroundImage(_foreground, _foregroundScale);
  }
}

void VisualizationWorkstationExtensionPlugin::onOpacityChanged(double opacity) {
  if (_viewer) {
    _viewer->setForegroundOpacity(opacity);
    _opacity = opacity;
  }
}

void VisualizationWorkstationExtensionPlugin::onEnableSegmentationToggled(bool toggled) {
  if (!toggled) {
    removeSegmentationsFromViewer();
  }
  else {
    addSegmentationsToViewer();
  }
}

void VisualizationWorkstationExtensionPlugin::addSegmentationsToViewer() {
  if (_lst) {
    std::vector<Annotation*> tmp = _lst->getAnnotations();
    float scl = _viewer->getSceneScale();
    for (std::vector<Annotation*>::iterator it = tmp.begin(); it != tmp.end(); ++it) {
      QPolygonF poly;
      std::vector<Point> coords = (*it)->getCoordinates();
      for (std::vector<Point>::iterator pt = coords.begin(); pt != coords.end(); ++pt) {
        poly.append(QPointF(pt->getX()*scl, pt->getY()*scl));
      }
      QGraphicsPolygonItem* cur = new QGraphicsPolygonItem(poly);
      cur->setBrush(QBrush());
      cur->setPen(QPen(QBrush(QColor("red")), 1.));
      _viewer->scene()->addItem(cur);
      cur->setZValue(std::numeric_limits<float>::max());
      _polygons.append(cur);
    }
  }
}

void VisualizationWorkstationExtensionPlugin::removeSegmentationsFromViewer() {
  if (!_polygons.empty()) {
    for (QList<QGraphicsPolygonItem*>::iterator it = _polygons.begin(); it != _polygons.end(); ++it) {
      _viewer->scene()->removeItem(*it);
      delete (*it);
    }
    _polygons.clear();
  }
}