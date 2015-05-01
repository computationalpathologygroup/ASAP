#include "VisualizationWorkstationExtensionPlugin.h"
#include "../PathologyViewer.h"
#include <QDockWidget>
#include <QtUiTools>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include "io/multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "core/filetools.h"

VisualizationWorkstationExtensionPlugin::VisualizationWorkstationExtensionPlugin() :
  WorkstationExtensionPluginInterface(),
  _dockWidget(NULL),
  _likelihoodCheckBox(NULL),
  _foreground(NULL)
{

}

VisualizationWorkstationExtensionPlugin::~VisualizationWorkstationExtensionPlugin() {
  if (_foreground) {
    emit changeForegroundImage(NULL);
    _foreground = NULL;
  }
  _dockWidget = NULL;
}

bool VisualizationWorkstationExtensionPlugin::initialize(PathologyViewer* viewer) {
  _viewer = viewer;
  connect(this, SIGNAL(changeForegroundImage(MultiResolutionImage*)), viewer, SLOT(onForegroundImageChanged(MultiResolutionImage*)));
  return true;
}

QDockWidget* VisualizationWorkstationExtensionPlugin::getDockWidget() {
  _dockWidget = new QDockWidget("Cancer detection visualization");
  QUiLoader loader;
  QFile file(":/VisualizationWorkstationExtensionPlugin.ui");
  file.open(QFile::ReadOnly);
  QWidget* content = loader.load(&file, _dockWidget);
  file.close();
  _likelihoodCheckBox = content->findChild<QCheckBox*>("LikelihoodCheckBox");
  QDoubleSpinBox* spinBox = content->findChild<QDoubleSpinBox*>("OpacitySpinBox");
  connect(_likelihoodCheckBox, SIGNAL(toggled(bool)), this, SLOT(onEnableLikelihoodToggled(bool)));
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
    std::string likImgPth = core::completePath(base + "_likelihood.tif", core::extractFilePath(fileName));
    if (core::fileExists(likImgPth)) {
      MultiResolutionImageReader reader;
      if (_foreground) {
        emit changeForegroundImage(NULL);
        _foreground = NULL;
      }
      _foreground = reader.open(likImgPth);
      if (_foreground && _likelihoodCheckBox && _likelihoodCheckBox->isChecked()) {
        emit changeForegroundImage(_foreground);
      }
      else {
        emit changeForegroundImage(NULL);
      }
    }
  }
}

void VisualizationWorkstationExtensionPlugin::onImageClosed() {
  if (_foreground) {
    emit changeForegroundImage(NULL);
    delete _foreground;
    _foreground = NULL;
  }
  if (_dockWidget) {
    _dockWidget->setEnabled(false);
  }
}

void VisualizationWorkstationExtensionPlugin::onEnableLikelihoodToggled(bool toggled) {
  if (!toggled) {
    emit changeForegroundImage(NULL);
  }
  else {
    emit changeForegroundImage(_foreground);
  }
}

void VisualizationWorkstationExtensionPlugin::onOpacityChanged(double opacity) {
  if (_viewer) {
    _viewer->setForegroundOpacity(opacity);
  }
}

void VisualizationWorkstationExtensionPlugin::onEnableSegmentationToggled(bool toggled) {

}