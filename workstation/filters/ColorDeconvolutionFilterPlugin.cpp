#include "ColorDeconvolutionFilterPlugin.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include <iostream>

// Qt widget used for panel
#include <QDialog>
#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QDialog>
#include <QtUiTools>

Q_DECLARE_METATYPE(Patch<double>*)

ColorDeconvolutionFilterPlugin::ColorDeconvolutionFilterPlugin() :
  ImageFilterPluginInterface()
{
  _filter = new ColorDeconvolutionFilter<double>();
}

bool ColorDeconvolutionFilterPlugin::initialize(const ImageSource* img) {
  if (img->getColorType() == pathology::ColorType::RGB || img->getColorType() == pathology::ColorType::ARGB) {
    return true;
  }
  else {
    return false;
  }
}

ImageFilterPluginInterface* ColorDeconvolutionFilterPlugin::clone() const {
  ColorDeconvolutionFilterPlugin* clone = new ColorDeconvolutionFilterPlugin(*this);
  return clone;
}

ColorDeconvolutionFilterPlugin::ColorDeconvolutionFilterPlugin(const ColorDeconvolutionFilterPlugin& other) :
  ImageFilterPluginInterface()
{
  _mutex.lock();
  if (_filter) {
    delete _filter;
    _filter = NULL;
  }
  ColorDeconvolutionFilter<double>* otherFilter = dynamic_cast<ColorDeconvolutionFilter<double>* >(other._filter);
  if (otherFilter) {
    _filter = new ColorDeconvolutionFilter<double>(*otherFilter);
  }
  _mutex.unlock();
}

QString ColorDeconvolutionFilterPlugin::name() const {
  return QString::fromStdString("Color Deconvolution");
}

QIcon ColorDeconvolutionFilterPlugin::icon() const {
  return QIcon(QPixmap(":/icons/ColorDeconvolutionIcon.png"));
}

void ColorDeconvolutionFilterPlugin::filter(const Patch<double> &input, QVariant &output) {
  Patch<double>* outImg = new Patch<double>();
  ColorDeconvolutionFilter<double>* filter = dynamic_cast<ColorDeconvolutionFilter<double>* >(_filter);
  if (filter) {
    if (filter->filter(input, *outImg)) {
      output = QVariant::fromValue<Patch<double>*>(outImg);
    }
    else {
      delete outImg;
    }
  }
}

QPointer<QWidget> ColorDeconvolutionFilterPlugin::getSettingsPanel() {
  initializeSettingsPanel();
  updateSettingsPanelFromFilter();
  return _settingsPanel;
}

void ColorDeconvolutionFilterPlugin::initializeSettingsPanel() {
  _mutex.lock();
  if (_settingsPanel) {
    _settingsPanel->deleteLater();
  }
  QUiLoader loader;
  QFile file(":/ColorDeconvolutionFilter.ui");
  file.open(QFile::ReadOnly);
  _settingsPanel = loader.load(&file);
  file.close();
  QDoubleSpinBox* stain1R = _settingsPanel->findChild<QDoubleSpinBox*>("Stain1RSpinBox");
  QDoubleSpinBox* stain1G = _settingsPanel->findChild<QDoubleSpinBox*>("Stain1GSpinBox");
  QDoubleSpinBox* stain1B = _settingsPanel->findChild<QDoubleSpinBox*>("Stain1BSpinBox");
  QDoubleSpinBox* stain2R = _settingsPanel->findChild<QDoubleSpinBox*>("Stain2RSpinBox");
  QDoubleSpinBox* stain2G = _settingsPanel->findChild<QDoubleSpinBox*>("Stain2GSpinBox");
  QDoubleSpinBox* stain2B = _settingsPanel->findChild<QDoubleSpinBox*>("Stain2BSpinBox");
  QDoubleSpinBox* stain3R = _settingsPanel->findChild<QDoubleSpinBox*>("Stain3RSpinBox");
  QDoubleSpinBox* stain3G = _settingsPanel->findChild<QDoubleSpinBox*>("Stain3GSpinBox");
  QDoubleSpinBox* stain3B = _settingsPanel->findChild<QDoubleSpinBox*>("Stain3BSpinBox");
  QDoubleSpinBox* rThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("RThreshold");
  QDoubleSpinBox* gThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("GThreshold");
  QDoubleSpinBox* bThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("BThreshold");
  QDoubleSpinBox* globalThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("GlobalThreshold");
  QPushButton* revertStainButton = _settingsPanel->findChild<QPushButton*>("RevertStainsToDefaultButton");
  QComboBox* outputStain = _settingsPanel->findChild<QComboBox*>("OutputStainComboBox");
  connect(stain1R, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(stain1G, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(stain1B, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(stain2R, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(stain2G, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(stain2B, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(stain3R, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(stain3G, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(stain3B, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(rThreshold, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(gThreshold, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(bThreshold, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(globalThreshold, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(outputStain, SIGNAL(currentIndexChanged(int)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(revertStainButton, SIGNAL(clicked()), this, SLOT(revertStainToDefault()));
  _mutex.unlock();
}

void ColorDeconvolutionFilterPlugin::revertStainToDefault() {
  if (_filter) {
    ColorDeconvolutionFilter<double>* filter = dynamic_cast<ColorDeconvolutionFilter<double>* >(_filter);
    if (filter) {
      filter->revertToDefaultStain();
      updateSettingsPanelFromFilter();
      emit filterParametersChanged();
    }
  }
}

void ColorDeconvolutionFilterPlugin::updateFilterFromSettingsPanel() {
  ColorDeconvolutionFilter<double>* filter = dynamic_cast<ColorDeconvolutionFilter<double>* >(_filter);
  if (_settingsPanel && filter) {
    QDoubleSpinBox* stain1R = _settingsPanel->findChild<QDoubleSpinBox*>("Stain1RSpinBox");
    QDoubleSpinBox* stain1G = _settingsPanel->findChild<QDoubleSpinBox*>("Stain1GSpinBox");
    QDoubleSpinBox* stain1B = _settingsPanel->findChild<QDoubleSpinBox*>("Stain1BSpinBox");
    QDoubleSpinBox* stain2R = _settingsPanel->findChild<QDoubleSpinBox*>("Stain2RSpinBox");
    QDoubleSpinBox* stain2G = _settingsPanel->findChild<QDoubleSpinBox*>("Stain2GSpinBox");
    QDoubleSpinBox* stain2B = _settingsPanel->findChild<QDoubleSpinBox*>("Stain2BSpinBox");
    QDoubleSpinBox* stain3R = _settingsPanel->findChild<QDoubleSpinBox*>("Stain3RSpinBox");
    QDoubleSpinBox* stain3G = _settingsPanel->findChild<QDoubleSpinBox*>("Stain3GSpinBox");
    QDoubleSpinBox* stain3B = _settingsPanel->findChild<QDoubleSpinBox*>("Stain3BSpinBox");
    QDoubleSpinBox* rThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("RThreshold");
    QDoubleSpinBox* gThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("GThreshold");
    QDoubleSpinBox* bThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("BThreshold");
    QDoubleSpinBox* globalThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("GlobalThreshold");
    QComboBox* outputStain = _settingsPanel->findChild<QComboBox*>("OutputStainComboBox");
    filter->setGlobalDensityThreshold(globalThreshold->value());
    std::vector<double> rgbThresholds;
    rgbThresholds.push_back(rThreshold->value());
    rgbThresholds.push_back(gThreshold->value());
    rgbThresholds.push_back(bThreshold->value());
    filter->setRGBDensityThresholds(rgbThresholds);
    std::vector<std::vector<double> > stains(3, std::vector<double>(3, 0));
    stains[0][0] = stain1R->value();
    stains[0][1] = stain1G->value();
    stains[0][2] = stain1B->value();
    stains[1][0] = stain2R->value();
    stains[1][1] = stain2G->value();
    stains[1][2] = stain2B->value();
    stains[2][0] = stain3R->value();
    stains[2][1] = stain3G->value();
    stains[2][2] = stain3B->value();
    filter->setStain(stains[0], stains[1], stains[2]);
    filter->setOutputStain(outputStain->currentIndex());
    emit filterParametersChanged();
  }
}

void ColorDeconvolutionFilterPlugin::updateSettingsPanelFromFilter() {
  ColorDeconvolutionFilter<double>* filter = dynamic_cast<ColorDeconvolutionFilter<double>* >(_filter);
  if (_settingsPanel && filter) {
    _mutex.lock();
    QDoubleSpinBox* stain1R = _settingsPanel->findChild<QDoubleSpinBox*>("Stain1RSpinBox");
    QDoubleSpinBox* stain1G = _settingsPanel->findChild<QDoubleSpinBox*>("Stain1GSpinBox");
    QDoubleSpinBox* stain1B = _settingsPanel->findChild<QDoubleSpinBox*>("Stain1BSpinBox");
    QDoubleSpinBox* stain2R = _settingsPanel->findChild<QDoubleSpinBox*>("Stain2RSpinBox");
    QDoubleSpinBox* stain2G = _settingsPanel->findChild<QDoubleSpinBox*>("Stain2GSpinBox");
    QDoubleSpinBox* stain2B = _settingsPanel->findChild<QDoubleSpinBox*>("Stain2BSpinBox");
    QDoubleSpinBox* stain3R = _settingsPanel->findChild<QDoubleSpinBox*>("Stain3RSpinBox");
    QDoubleSpinBox* stain3G = _settingsPanel->findChild<QDoubleSpinBox*>("Stain3GSpinBox");
    QDoubleSpinBox* stain3B = _settingsPanel->findChild<QDoubleSpinBox*>("Stain3BSpinBox");
    QDoubleSpinBox* rThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("RThreshold");
    QDoubleSpinBox* gThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("GThreshold");
    QDoubleSpinBox* bThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("BThreshold");
    QDoubleSpinBox* globalThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("GlobalThreshold");
    QComboBox* outputStain = _settingsPanel->findChild<QComboBox*>("OutputStainComboBox");
    std::vector<std::vector<double> > stains = filter->getStain();
    std::vector<double> rgbThresholds = filter->getRGBDensityThresholds();
    double gbThreshold = filter->getGlobalDensityThreshold();
    unsigned int outStain = filter->getOutputStain();
    QList<QWidget*> boxes = _settingsPanel->findChildren<QWidget*>();
    foreach(QWidget* sb, boxes) {
      sb->blockSignals(true);
    }
    stain1R->setValue(stains[0][0]);
    stain1G->setValue(stains[0][1]);
    stain1B->setValue(stains[0][2]);
    stain2R->setValue(stains[1][0]);
    stain2G->setValue(stains[1][1]);
    stain2B->setValue(stains[1][2]);
    stain3R->setValue(stains[2][0]);
    stain3G->setValue(stains[2][1]);
    stain3B->setValue(stains[2][2]);
    rThreshold->setValue(rgbThresholds[0]);
    gThreshold->setValue(rgbThresholds[1]);
    bThreshold->setValue(rgbThresholds[2]);
    globalThreshold->setValue(gbThreshold);
    outputStain->setCurrentIndex(outStain);
    foreach(QWidget* sb, boxes) {
      sb->blockSignals(false);
    }
    _mutex.unlock();
  }
}