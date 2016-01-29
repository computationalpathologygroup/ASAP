#include "NucleiDetectionFilterPlugin.h"
#include "imgproc/opencv/NucleiDetectionFilter.h"
#include "core/Point.h"
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

Q_DECLARE_METATYPE(std::vector<Point>)

NucleiDetectionFilterPlugin::NucleiDetectionFilterPlugin() : 
  ImageFilterPluginInterface(),
  _monochromeInput(false)
{
  _filter.reset(new NucleiDetectionFilter<double>());
}

ImageFilterPluginInterface* NucleiDetectionFilterPlugin::clone() const {
  NucleiDetectionFilterPlugin* clone = new NucleiDetectionFilterPlugin(*this);
  return clone;
}

NucleiDetectionFilterPlugin::NucleiDetectionFilterPlugin(const NucleiDetectionFilterPlugin& other) :
  ImageFilterPluginInterface()
{
  const NucleiDetectionFilter<double>* otherFilter = dynamic_cast<const NucleiDetectionFilter<double>* >(other._filter.get());
  if (otherFilter) {
    _filter.reset(new NucleiDetectionFilter<double>(*otherFilter));
    initializeSettingsPanel();
    updateSettingsPanelFromFilter();
  }
}

QString NucleiDetectionFilterPlugin::name() const {
  return _filter ? QString::fromStdString("Nuclei Detection") : QString();
}

QIcon NucleiDetectionFilterPlugin::icon() const {
  return QIcon(QPixmap(":/NucleiDetectionFilter_icons/NucleiDetectionIcon.png"));
}

bool NucleiDetectionFilterPlugin::initialize(const ImageSource* img) {
  if (img->getColorType() == pathology::ColorType::RGB || img->getColorType() == pathology::ColorType::ARGB || img->getColorType() == pathology::ColorType::Monochrome) {
    if (img->getColorType() == pathology::ColorType::Monochrome) {
      _monochromeInput = true;
    }
    else {
      _monochromeInput = false;
    }
    return true;
  }
  else {
    return false;
  }
}

void NucleiDetectionFilterPlugin::filter(const Patch<double> &input, QVariant &output) {
  NucleiDetectionFilter<double>* filter = dynamic_cast<NucleiDetectionFilter<double>* >(_filter.get());
  if (filter) {
    _mutex.lock();
    if (_settingsPanel) {
      QLabel* nrDetectedNuclei = _settingsPanel->findChild<QLabel*>("NrOfDetectedNucleiValueLabel");
      if (nrDetectedNuclei) {
        nrDetectedNuclei->setText("");
      }
    }
    std::vector<Point> result;
    if (filter->filter(input, result)) {
      output = QVariant::fromValue<std::vector<Point> >(result);
      if (_settingsPanel) {
        QLabel* nrDetectedNuclei = _settingsPanel->findChild<QLabel*>("NrOfDetectedNucleiValueLabel");
        if (nrDetectedNuclei) {
          unsigned int detectedNuclei = filter->getNumberOfDetectedNuclei();
          nrDetectedNuclei->setText(QString::number(detectedNuclei));
        }
      }
    }
    else {
      if (_settingsPanel) {
        QLabel* nrDetectedNuclei = _settingsPanel->findChild<QLabel*>("NrOfDetectedNucleiValueLabel");
        if (nrDetectedNuclei) {
          unsigned int detectedNuclei = filter->getNumberOfDetectedNuclei();
          nrDetectedNuclei->setText("");
        }
      }
    }
    _mutex.unlock();
  }
}

QPointer<QWidget> NucleiDetectionFilterPlugin::getSettingsPanel() {
  initializeSettingsPanel();
  updateSettingsPanelFromFilter();
  return _settingsPanel;
}

void NucleiDetectionFilterPlugin::initializeSettingsPanel() {
  _mutex.lock();
  if (_settingsPanel) {
    _settingsPanel->deleteLater();
  }
  QUiLoader loader;
  QFile file(":/NucleiDetectionFilter_ui/NucleiDetectionFilter.ui");
  file.open(QFile::ReadOnly);
  _settingsPanel = loader.load(&file);
  file.close();
  QPushButton* revertStainButton = _settingsPanel->findChild<QPushButton*>("RevertStainsToDefaultButton");
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
  QDoubleSpinBox* maxRadius = _settingsPanel->findChild<QDoubleSpinBox*>("MaxRadiusSpinBox");
  QDoubleSpinBox* minRadius = _settingsPanel->findChild<QDoubleSpinBox*>("MinRadiusSpinBox");
  QDoubleSpinBox* stepRadius = _settingsPanel->findChild<QDoubleSpinBox*>("StepRadiusSpinBox");
  QDoubleSpinBox* alpha = _settingsPanel->findChild<QDoubleSpinBox*>("AlphaSpinBox");
  QDoubleSpinBox* beta = _settingsPanel->findChild<QDoubleSpinBox*>("BetaSpinBox");
  QDoubleSpinBox* hMaximaThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("HMaximaThresholdSpinBox");
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
  connect(maxRadius, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(minRadius, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(stepRadius, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(alpha, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(beta, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(hMaximaThreshold, SIGNAL(valueChanged(double)), this, SLOT(updateFilterFromSettingsPanel()));
  connect(revertStainButton, SIGNAL(clicked()), this, SLOT(revertStainToDefault()));
  QGroupBox* colorDeconvBox = _settingsPanel->findChild<QGroupBox*>("ColorDeconvolutionBox");
  if (_monochromeInput) {
    colorDeconvBox->setEnabled(false);
  }
  else {
    colorDeconvBox->setEnabled(true);
  }
  _mutex.unlock();
}

void NucleiDetectionFilterPlugin::revertStainToDefault() {
  NucleiDetectionFilter<double>* filter = dynamic_cast<NucleiDetectionFilter<double>* >(_filter.get());
  if (filter) {
    filter->getColorDeconvolutionFilter()->revertToDefaultStain();
    updateSettingsPanelFromFilter();
  }
}

void NucleiDetectionFilterPlugin::updateFilterFromSettingsPanel() {
  NucleiDetectionFilter<double>* filter = dynamic_cast<NucleiDetectionFilter<double>* >(_filter.get());
  if (_settingsPanel && filter) {
    filter->cancel();
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
    QDoubleSpinBox* maxRadius = _settingsPanel->findChild<QDoubleSpinBox*>("MaxRadiusSpinBox");
    QDoubleSpinBox* minRadius = _settingsPanel->findChild<QDoubleSpinBox*>("MinRadiusSpinBox");
    QDoubleSpinBox* stepRadius = _settingsPanel->findChild<QDoubleSpinBox*>("StepRadiusSpinBox");
    QDoubleSpinBox* alpha = _settingsPanel->findChild<QDoubleSpinBox*>("AlphaSpinBox");
    QDoubleSpinBox* beta = _settingsPanel->findChild<QDoubleSpinBox*>("BetaSpinBox");
    QDoubleSpinBox* hMaximaThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("HMaximaThresholdSpinBox");
    QLabel* nrDetectedNuclei = _settingsPanel->findChild<QLabel*>("NrOfDetectedNucleiValueLabel");
    filter->getColorDeconvolutionFilter()->setGlobalDensityThreshold(globalThreshold->value());
    std::vector<double> rgbThresholds;
    rgbThresholds.push_back(rThreshold->value());
    rgbThresholds.push_back(gThreshold->value());
    rgbThresholds.push_back(bThreshold->value());
    filter->getColorDeconvolutionFilter()->setRGBDensityThresholds(rgbThresholds);
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
    filter->getColorDeconvolutionFilter()->setStain(stains[0], stains[1], stains[2]);
    filter->setBeta(beta->value());
    filter->setAlpha(alpha->value());
    filter->setHMaximaThreshold(hMaximaThreshold->value());
    filter->setMaximumRadius(maxRadius->value());
    filter->setMinimumRadius(minRadius->value());
    filter->setRadiusStep(stepRadius->value());
    _mutex.unlock();
    emit filterParametersChanged();
  }
}

void NucleiDetectionFilterPlugin::updateSettingsPanelFromFilter() {
  NucleiDetectionFilter<double>* filter = dynamic_cast<NucleiDetectionFilter<double>* >(_filter.get());
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
    QDoubleSpinBox* maxRadius = _settingsPanel->findChild<QDoubleSpinBox*>("MaxRadiusSpinBox");
    QDoubleSpinBox* minRadius = _settingsPanel->findChild<QDoubleSpinBox*>("MinRadiusSpinBox");
    QDoubleSpinBox* stepRadius = _settingsPanel->findChild<QDoubleSpinBox*>("StepRadiusSpinBox");
    QDoubleSpinBox* alpha = _settingsPanel->findChild<QDoubleSpinBox*>("AlphaSpinBox");
    QDoubleSpinBox* beta = _settingsPanel->findChild<QDoubleSpinBox*>("BetaSpinBox");
    QDoubleSpinBox* hMaximaThreshold = _settingsPanel->findChild<QDoubleSpinBox*>("HMaximaThresholdSpinBox");
    
    std::vector<std::vector<double> > stains = filter->getColorDeconvolutionFilter()->getStain();
    std::vector<double> rgbThresholds = filter->getColorDeconvolutionFilter()->getRGBDensityThresholds();
    double gbThreshold = filter->getColorDeconvolutionFilter()->getGlobalDensityThreshold();
    float minRadiusValue = filter->getMinimumRadius();
    float maxRadiusValue = filter->getMaximumRadius();
    float stepRadiusValue = filter->getRadiusStep();
    float alphaValue = filter->getAlpha();
    float betaValue = filter->getBeta();
    float hMaximaThresholdValue = filter->getHMaximaThreshold();

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
    minRadius->setValue(minRadiusValue);
    maxRadius->setValue(maxRadiusValue);
    stepRadius->setValue(stepRadiusValue);
    alpha->setValue(alphaValue);
    beta->setValue(betaValue);
    hMaximaThreshold->setValue(hMaximaThresholdValue);
    foreach(QWidget* sb, boxes) {
      sb->blockSignals(false);
    }
    _mutex.unlock();
  }
}