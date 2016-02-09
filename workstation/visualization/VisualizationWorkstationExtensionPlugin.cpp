#include "VisualizationWorkstationExtensionPlugin.h"
#include "../PathologyViewer.h"
#include <QDockWidget>
#include <QtUiTools>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGraphicsPolygonItem>
#include <QPolygonF>
#include <QPushButton>
#include <QFrame>
#include <QGroupBox>
#include <QComboBox>
#include <QPushButton>
#include <QFileDialog>
#include "io/multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "core/filetools.h"
#include "core/PathologyEnums.h"
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
  _window(1.0),
  _level(0.5),
  _foregroundChannel(0)
{
  _lst = std::make_shared<AnnotationList>();
  _xmlRepo = std::make_shared<XmlRepository>(_lst);
  _settings = new QSettings(QSettings::IniFormat, QSettings::UserScope, "DIAG", "ASAP", this);
}

VisualizationWorkstationExtensionPlugin::~VisualizationWorkstationExtensionPlugin() { 
  if (_foreground) {
    _foregroundScale = 1.;
    emit changeForegroundImage(std::weak_ptr<MultiResolutionImage>(), _foregroundScale);
    _foreground.reset();
  }
  _dockWidget = NULL;
}

bool VisualizationWorkstationExtensionPlugin::initialize(PathologyViewer* viewer) {
  _viewer = viewer;
  connect(this, SIGNAL(changeForegroundImage(std::weak_ptr<MultiResolutionImage>, float)), viewer, SLOT(onForegroundImageChanged(std::weak_ptr<MultiResolutionImage>, float)));
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
  spinBox->setValue(_opacity);
  QDoubleSpinBox* windowSpinBox = content->findChild<QDoubleSpinBox*>("WindowSpinBox");
  windowSpinBox->setValue(_window);
  QDoubleSpinBox* levelSpinBox = content->findChild<QDoubleSpinBox*>("LevelSpinBox");
  levelSpinBox->setValue(_level);
  QSpinBox* channelSpinBox = content->findChild<QSpinBox*>("ChannelSpinBox");
  channelSpinBox->setValue(_foregroundChannel);
  _segmentationCheckBox = content->findChild<QCheckBox*>("SegmentationCheckBox");
  QPushButton* openResultButton = content->findChild<QPushButton*>("OpenResultPushButton");
  QComboBox* LUTBox = content->findChild<QComboBox*>("LUTComboBox");
  LUTBox->setEditable(false);
  for (std::map<std::string, pathology::LUT>::const_iterator it = pathology::ColorLookupTables.begin(); it != pathology::ColorLookupTables.end(); ++it) {
    LUTBox->addItem(QString::fromStdString(it->first));
  }
  LUTBox->setCurrentText("Normal");
  connect(_likelihoodCheckBox, SIGNAL(toggled(bool)), this, SLOT(onEnableLikelihoodToggled(bool)));
  connect(_segmentationCheckBox, SIGNAL(toggled(bool)), this, SLOT(onEnableSegmentationToggled(bool)));
  connect(spinBox, SIGNAL(valueChanged(double)), this, SLOT(onOpacityChanged(double)));
  connect(windowSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onWindowValueChanged(double)));
  connect(levelSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onLevelValueChanged(double)));
  connect(channelSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onChannelChanged(int)));
  connect(openResultButton, SIGNAL(clicked()), this, SLOT(onOpenResultImageClicked()));
  connect(LUTBox, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(onLUTChanged(const QString&)));
  _dockWidget->setEnabled(false);

  // Not used for know
  QGroupBox* segmentationGroupBox = content->findChild<QGroupBox*>("SegmentationGroupBox");
  segmentationGroupBox->setVisible(false);
  return _dockWidget;
}

void VisualizationWorkstationExtensionPlugin::onNewImageLoaded(std::weak_ptr<MultiResolutionImage> img, std::string fileName) {
  std::shared_ptr<MultiResolutionImage> local_img = img.lock();
  _backgroundDimensions = local_img->getDimensions();
  if (_dockWidget) {
    _dockWidget->setEnabled(true);
  }
  if (!fileName.empty()) {
    std::string base = core::extractBaseName(fileName);
    std::string likImgPth = core::completePath(base + "_likelihood_map.tif", core::extractFilePath(fileName));
    std::string segmXMLPth = core::completePath(base + "_detections.xml", core::extractFilePath(fileName));
    this->loadNewForegroundImage(likImgPth);
    if (core::fileExists(segmXMLPth)) {
      _xmlRepo->setSource(segmXMLPth);
      _xmlRepo->load();
      if (_segmentationCheckBox && _segmentationCheckBox->isChecked()) {
        addSegmentationsToViewer();
      }
    }
  }
}

void VisualizationWorkstationExtensionPlugin::onOpenResultImageClicked() {
  QString fileName = QFileDialog::getOpenFileName(_dockWidget, tr("Open File"), _settings->value("lastOpenendPath", QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)).toString(), tr("Result files (*.tif);;All files (*.*)"));
  if (!fileName.isEmpty()) {
    loadNewForegroundImage(fileName.toStdString());
  }
}

void VisualizationWorkstationExtensionPlugin::loadNewForegroundImage(const std::string& resultImagePth) {
  if (_foreground) {
    _foregroundScale = 1;
    emit changeForegroundImage(std::weak_ptr<MultiResolutionImage>(), _foregroundScale);
    _foreground.reset();
  }
  QGroupBox* segmentationGroupBox = _dockWidget->findChild<QGroupBox*>("SegmentationGroupBox");
  segmentationGroupBox->setEnabled(false);
  QGroupBox* visualizationGroupBox = _dockWidget->findChild<QGroupBox*>("VisualizationGroupBox");
  visualizationGroupBox->setEnabled(false);
  if (core::fileExists(resultImagePth)) {
    MultiResolutionImageReader reader;
    _foreground.reset(reader.open(resultImagePth));
    if (_foreground) {
      setDefaultVisualizationParameters(_foreground);
      std::vector<unsigned long long> dimsFG = _foreground->getDimensions();
      if (_backgroundDimensions[0] / dimsFG[0] == _backgroundDimensions[1] / dimsFG[1]) {
        _foregroundScale = _backgroundDimensions[0] / dimsFG[0];
        if (_likelihoodCheckBox && _likelihoodCheckBox->isChecked()) {
          emit changeForegroundImage(_foreground, _foregroundScale);
        }
        else {
          emit changeForegroundImage(std::weak_ptr<MultiResolutionImage>(), _foregroundScale);
        }
      }
      QGroupBox* segmentationGroupBox = _dockWidget->findChild<QGroupBox*>("SegmentationGroupBox");
      segmentationGroupBox->setEnabled(true);
      QGroupBox* visualizationGroupBox = _dockWidget->findChild<QGroupBox*>("VisualizationGroupBox");
      visualizationGroupBox->setEnabled(true);
    }
  }
}

void VisualizationWorkstationExtensionPlugin::setDefaultVisualizationParameters(std::shared_ptr<MultiResolutionImage> img) {
  if (_dockWidget) {
    _opacity = 0.5;
    QDoubleSpinBox* spinBox = _dockWidget->findChild<QDoubleSpinBox*>("OpacitySpinBox");
    spinBox->setValue(_opacity);
    _viewer->setForegroundOpacity(_opacity);
    _foregroundChannel = 0;
    QSpinBox* channelSpinBox = _dockWidget->findChild<QSpinBox*>("ChannelSpinBox");
    channelSpinBox->setValue(_foregroundChannel);
    channelSpinBox->setMaximum(_foreground->getSamplesPerPixel() - 1);
    _viewer->setForegroundChannel(_foregroundChannel);

    double maxValue = img->getMaxValue(_foregroundChannel);
    double minValue = img->getMinValue(_foregroundChannel);
    _window = maxValue - minValue;
    QDoubleSpinBox* windowSpinBox = _dockWidget->findChild<QDoubleSpinBox*>("WindowSpinBox");
    windowSpinBox->setValue(_window);
    _level = (maxValue + minValue) / 2;
    QDoubleSpinBox* levelSpinBox = _dockWidget->findChild<QDoubleSpinBox*>("LevelSpinBox");
    levelSpinBox->setValue(_level);
    _viewer->setForegroundWindowAndLevel(_window, _level);

    QComboBox* LUTBox = _dockWidget->findChild<QComboBox*>("LUTComboBox");
    if (img->getDataType() == pathology::UChar || img->getDataType() == pathology::UInt32 || img->getDataType() == pathology::UInt16) {
      LUTBox->setCurrentText("Label");
      _currentLUT = "Label";
      _viewer->setForegroundLUT("Label");
    }
    else {
      LUTBox->setCurrentText("Traffic Light");
      _currentLUT = "Traffic Light";
      _viewer->setForegroundLUT("Traffic Light");
    }
  }
}

void VisualizationWorkstationExtensionPlugin::onImageClosed() {
  if (!_polygons.empty()) {
    removeSegmentationsFromViewer();
  }
  if (_foreground) {
    _foregroundScale = 1;
    emit changeForegroundImage(std::weak_ptr<MultiResolutionImage>(), _foregroundScale);
    _foreground.reset();
  }
  if (_dockWidget) {
    _dockWidget->setEnabled(false);
    QGroupBox* segmentationGroupBox = _dockWidget->findChild<QGroupBox*>("SegmentationGroupBox");
    segmentationGroupBox->setEnabled(false);
    QGroupBox* visualizationGroupBox = _dockWidget->findChild<QGroupBox*>("VisualizationGroupBox");
    visualizationGroupBox->setEnabled(false);
  }
}

void VisualizationWorkstationExtensionPlugin::onEnableLikelihoodToggled(bool toggled) {
  if (!toggled) {
    emit changeForegroundImage(std::weak_ptr<MultiResolutionImage>(), _foregroundScale);
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

void VisualizationWorkstationExtensionPlugin::onLUTChanged(const QString& LUTname) {
  if (_viewer && LUTname != _currentLUT) {
    _currentLUT = LUTname;
    _viewer->setForegroundLUT(LUTname.toStdString());
  }
}
void VisualizationWorkstationExtensionPlugin::onWindowValueChanged(double window) {
  if (_viewer && window != _window) {
    _window = window;
    _viewer->setForegroundWindowAndLevel(_window, _level);
  }
}
void VisualizationWorkstationExtensionPlugin::onLevelValueChanged(double level) {
  if (_viewer && level != _level) {
    _level = level;
    _viewer->setForegroundWindowAndLevel(_window, _level);
  }
}
void VisualizationWorkstationExtensionPlugin::onChannelChanged(int channel) {
  if (_viewer && channel != _foregroundChannel) {
    _foregroundChannel = channel;
    _viewer->setForegroundChannel(_foregroundChannel);
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
    std::vector<std::shared_ptr<Annotation> > tmp = _lst->getAnnotations();
    float scl = _viewer->getSceneScale();
    for (std::vector<std::shared_ptr<Annotation> >::iterator it = tmp.begin(); it != tmp.end(); ++it) {
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