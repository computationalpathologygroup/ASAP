#include "VisualizationWorkstationExtensionPlugin.h"
#include <iostream>
#include "../PathologyViewer.h"
#include <QDockWidget>
#include <QtUiTools>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGraphicsPolygonItem>
#include <QPolygonF>
#include <QDialog>
#include <QPushButton>
#include <QFrame>
#include <QGroupBox>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QMessageBox>
#include <QFileDialog>
#include <QToolButton>
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "core/filetools.h"
#include "core/PathologyEnums.h"
#include "annotation/XmlRepository.h"
#include "annotation/Annotation.h"
#include "annotation/AnnotationList.h"

VisualizationWorkstationExtensionPlugin::VisualizationWorkstationExtensionPlugin() :
  WorkstationExtensionPluginInterface(),
  _dockWidget(NULL),
  _LUTEditor(NULL),
  _likelihoodCheckBox(NULL),
  _foreground(NULL),
  _foregroundScale(1.),
  _opacity(1.0),
  _foregroundChannel(0),
  _renderingEnabled(false),
  _editingLUT(false),
  _previewingLUT(false)
{
  qRegisterMetaTypeStreamOperators<QList<float>>("QListFloat");
  qRegisterMetaTypeStreamOperators<QList<QList<float>>>("QListRGBAArray");

  _settings = new QSettings(QSettings::IniFormat, QSettings::UserScope, "DIAG", "ASAP", this);
  _settings->beginGroup("VisualizationWorkstationExtensionPlugin");
  int nrLUTs = _settings->beginReadArray("LUTs");
  if (nrLUTs > 0) {
    for (int i = 0; i < nrLUTs; ++i) {
      _settings->setArrayIndex(i);
      std::string lutName = _settings->value("lutName").value<QString>().toStdString();
      std::vector<float> lutIndices = _settings->value("lutIndices").value<QList<float>>().toVector().toStdVector();
      QList<QList<float>> qtLUTColors = _settings->value("lutColors").value<QList<QList<float>>>();
      std::vector<rgbaArray> lutColors;;
      for (auto color : qtLUTColors) {
        lutColors.push_back({ color[0], color[1], color[2], color[3] });
      }
      pathology::LUT lut = { lutIndices, lutColors };
      _colorLookupTables[lutName] = lut;
    }
  }
  else {
    _colorLookupTables = pathology::DefaultColorLookupTables;
  }
  _settings->endArray();
  _settings->endGroup();
}

VisualizationWorkstationExtensionPlugin::~VisualizationWorkstationExtensionPlugin() { 
  if (_settings) {
    _settings->beginGroup("VisualizationWorkstationExtensionPlugin");
    _settings->beginWriteArray("LUTs");
    int arrayIndex = 0;
    for (auto lut : _colorLookupTables) {
      _settings->setArrayIndex(arrayIndex);
      _settings->setValue("lutName", QString::fromStdString(lut.first));
      QList<float> indices = QList<float>::fromVector(QVector<float>::fromStdVector(lut.second.indices));
      const std::vector<rgbaArray>& colors = lut.second.colors;
      QList<QList<float>> qtColors;
      for (auto color : colors) {
        qtColors.append({ color[0], color[1], color[2], color[3] });
      }
      _settings->setValue("lutIndices", QVariant::fromValue<QList<float> >(indices));
      _settings->setValue("lutColors", QVariant::fromValue<QList<QList<float>>>(qtColors));
      arrayIndex++;
    }
    _settings->endArray();
    _settings->endGroup();
  }
  this->clearForegroundImage();
  _dockWidget = NULL;
}

bool VisualizationWorkstationExtensionPlugin::initialize(PathologyViewer* viewer) {
  _viewer = viewer;
  connect(this, SIGNAL(changeForegroundImage(std::weak_ptr<MultiResolutionImage>, float)), viewer, SLOT(onForegroundImageChanged(std::weak_ptr<MultiResolutionImage>, float)));
  return true;
}

QDockWidget* VisualizationWorkstationExtensionPlugin::getDockWidget() {
  _dockWidget = new QDockWidget("Overlays");
  QUiLoader loader;
  QFile mainGUIFile(":/VisualizationWorkstationExtensionPlugin_ui/VisualizationWorkstationExtensionPlugin.ui");
  mainGUIFile.open(QFile::ReadOnly);
  QWidget* content = loader.load(&mainGUIFile, _dockWidget);
  mainGUIFile.close();
  QFile LUTEditorFile(":/VisualizationWorkstationExtensionPlugin_ui/LUTEditor.ui");
  LUTEditorFile.open(QFile::ReadOnly);
  _LUTEditor = qobject_cast<QDialog*>(loader.load(&LUTEditorFile, NULL));
  LUTEditorFile.close();
  _likelihoodCheckBox = content->findChild<QCheckBox*>("LikelihoodCheckBox");
  QDoubleSpinBox* spinBox = content->findChild<QDoubleSpinBox*>("OpacitySpinBox");
  spinBox->setValue(_opacity);
  QSpinBox* channelSpinBox = content->findChild<QSpinBox*>("ChannelSpinBox");
  channelSpinBox->setValue(_foregroundChannel);
  QPushButton* openResultButton = content->findChild<QPushButton*>("OpenResultPushButton");
  QComboBox* LUTBox = content->findChild<QComboBox*>("LUTComboBox");
  QComboBox* LUTEditorBox = _LUTEditor->findChild<QComboBox*>("LUTListComboBox");
  QCheckBox* previewCheckBox = _LUTEditor->findChild<QCheckBox*>("previewCheckBox");
  QPushButton* addLUTButton = _LUTEditor->findChild<QPushButton*>("addLUTButton");
  QPushButton* removeLUTButton = _LUTEditor->findChild<QPushButton*>("removeLUTButton");
  QPushButton* duplicateLUTButton = _LUTEditor->findChild<QPushButton*>("duplicateLUTButton");
  QPushButton* resetAllLUTButton = _LUTEditor->findChild<QPushButton*>("resetAllLUTButton");
  previewCheckBox->setCheckState(Qt::Unchecked);
  LUTBox->setEditable(false);
  for (std::map<std::string, pathology::LUT>::const_iterator it = _colorLookupTables.begin(); it != _colorLookupTables.end(); ++it) {
    LUTBox->addItem(QString::fromStdString(it->first));
    LUTEditorBox->addItem(QString::fromStdString(it->first));
  }
  LUTBox->setCurrentText("Normal");
  LUTEditorBox->setCurrentText("Normal");
  this->generateLUTEditingWidgets("Normal");
  QToolButton* openLUTEditorButton = content->findChild<QToolButton*>("LUTEditorButton");
  connect(_likelihoodCheckBox, SIGNAL(toggled(bool)), this, SLOT(onEnableLikelihoodToggled(bool)));
  connect(spinBox, SIGNAL(valueChanged(double)), this, SLOT(onOpacityChanged(double)));
  connect(channelSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onChannelChanged(int)));
  connect(openResultButton, SIGNAL(clicked()), this, SLOT(onOpenResultImageClicked()));
  connect(addLUTButton, SIGNAL(clicked()), this, SLOT(addLUT()));
  connect(removeLUTButton, SIGNAL(clicked()), this, SLOT(removeLUT()));
  connect(duplicateLUTButton, SIGNAL(clicked()), this, SLOT(duplicateLUT()));
  connect(resetAllLUTButton, SIGNAL(clicked()), this, SLOT(resetAllLUTs()));
  connect(LUTBox, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(onLUTChanged(const QString&)));
  connect(LUTEditorBox, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(onLUTChanged(const QString&)));
  connect(openLUTEditorButton, SIGNAL(clicked()), this, SLOT(handleEditLUTRequest()));
  connect(previewCheckBox, SIGNAL(stateChanged(int)), this, SLOT(updateLUTPreviewStatus(int)));
  _dockWidget->setEnabled(false);

  return _dockWidget;
}

void VisualizationWorkstationExtensionPlugin::generateLUTEditingWidgets(const QString& currentLUTName) {
  pathology::LUT currentLUT = _colorLookupTables[currentLUTName.toStdString()];
  QScrollArea* scrollArea = _LUTEditor->findChild<QScrollArea*>("LUTEditorScrollArea");
  _LUTEditingArea = scrollArea->widget();
  delete _LUTEditingArea;
  _LUTEditingArea = new QWidget(_LUTEditor);
  _LUTEditingArea->setObjectName(QString("editingArea"));
  QVBoxLayout* vlayout = new QVBoxLayout;
  vlayout->setObjectName("vboxLUTLayout");
  _LUTEditingArea->setLayout(vlayout);
  for (int i = 0; i < currentLUT.indices.size(); ++i) {
    qobject_cast<QVBoxLayout*>(_LUTEditingArea->layout())->addLayout(createLUTEntry(currentLUT, i));
  }
  qobject_cast<QVBoxLayout*>(_LUTEditingArea->layout())->addStretch(1);
  scrollArea->setWidget(_LUTEditingArea);
}

QHBoxLayout* VisualizationWorkstationExtensionPlugin::createLUTEntry(const pathology::LUT& currentLUT, int index) {
  QHBoxLayout* editingEntry = new QHBoxLayout();
  editingEntry->setObjectName(QString("editingEntry_") + QString::number(index));
  QToolButton* colorSquare = new QToolButton();
  QDoubleSpinBox* value = new QDoubleSpinBox();
  value->setMinimum(-1000);
  value->setMaximum(1000);
  value->setObjectName(QString("value_") + QString::number(index));
  QToolButton* removeEntry = new QToolButton();;
  removeEntry->setObjectName(QString("removeButton_") + QString::number(index));
  connect(removeEntry, SIGNAL(clicked()), this, SLOT(removeLUTEntry()));
  QToolButton* addEntry = new QToolButton();;
  addEntry->setObjectName(QString("addButton_") + QString::number(index));
  connect(addEntry, SIGNAL(clicked()), this, SLOT(addLUTEntry()));
  colorSquare->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  colorSquare->setGeometry(QRect(0, 0, 25, 25));
  colorSquare->setObjectName(QString("color_") + QString::number(index));
  QPixmap square(25, 25);
  const rgbaArray& rgba = currentLUT.colors[index];
  square.fill(QColor(rgba[0], rgba[1], rgba[2], rgba[3]));
  colorSquare->setIcon(square);
  connect(colorSquare, SIGNAL(clicked()), this, SLOT(pickLUTColor()));
  value->setValue(currentLUT.indices[index]);
  connect(value, SIGNAL(valueChanged(double)), this, SLOT(onLUTIndexChanged(double)));
  removeEntry->setText("X");
  addEntry->setText("+");
  editingEntry->addWidget(colorSquare);
  editingEntry->addWidget(value);
  editingEntry->addWidget(removeEntry);
  editingEntry->addWidget(addEntry);
  return editingEntry;
}

void VisualizationWorkstationExtensionPlugin::pickLUTColor() {
  QToolButton* colorSquare = qobject_cast<QToolButton*>(sender());
  QPixmap old = colorSquare->icon().pixmap(25,25);
  QPixmap square(25, 25);
  QColor newColor = QColorDialog::getColor(old.toImage().pixelColor(0, 0), NULL, QString("Select a color"), QColorDialog::ShowAlphaChannel);
  square.fill(newColor);
  int red = newColor.red();
  int green = newColor.green();
  int blue = newColor.blue();
  int alpha = newColor.alpha();
  QString squareName = colorSquare->objectName();
  QStringList nameList = squareName.split("_");
  int colorIndex = nameList[1].toInt();
  rgbaArray newRGBA = { red, green, blue, alpha };
  _colorLookupTables[_currentLUT.toStdString()].colors[colorIndex] = newRGBA;
  colorSquare->setIcon(square);
  if (_editingLUT && _previewingLUT) {
    onLUTEntryChanged();
  }
}

void VisualizationWorkstationExtensionPlugin::onLUTIndexChanged(double newIndex) {
  QDoubleSpinBox* valueSpinBox = qobject_cast<QDoubleSpinBox*>(sender());
  QString spinBoxName = valueSpinBox->objectName();
  QStringList nameList = spinBoxName.split("_");
  int indexIndex = nameList[1].toInt();
  _colorLookupTables[_currentLUT.toStdString()].indices[indexIndex] = newIndex;
  if (_editingLUT && _previewingLUT) {
    onLUTEntryChanged();
  }
}

void VisualizationWorkstationExtensionPlugin::addLUTEntry() {
  QToolButton* addButton = qobject_cast<QToolButton*>(sender());
  QString addName = addButton->objectName();
  QStringList nameList = addName.split("_");
  int addIndex = nameList[1].toInt();

  float newValue = 999.;
  if (_colorLookupTables[_currentLUT.toStdString()].indices.size() > (addIndex + 1)) {
    newValue = (_colorLookupTables[_currentLUT.toStdString()].indices[addIndex] + _colorLookupTables[_currentLUT.toStdString()].indices[addIndex + 1]) / 2;
  }
  _colorLookupTables[_currentLUT.toStdString()].indices.insert(_colorLookupTables[_currentLUT.toStdString()].indices.begin() + addIndex + 1, newValue);
  rgbaArray newEntryColor = _colorLookupTables[_currentLUT.toStdString()].colors[addIndex];
  _colorLookupTables[_currentLUT.toStdString()].colors.insert(_colorLookupTables[_currentLUT.toStdString()].colors.begin() + addIndex + 1, newEntryColor);

  QVBoxLayout* parentLayout = qobject_cast<QVBoxLayout*>(qobject_cast<QWidget*>(addButton->parent())->layout());
  for (auto editEnty : parentLayout->children()) {
    QHBoxLayout* hLayout = qobject_cast<QHBoxLayout*>(editEnty);
    if (hLayout) {
      QString hlayoutName = hLayout->objectName();
      QStringList nameAndIndex = hlayoutName.split("_");
      int layoutIndex = nameAndIndex[1].toInt();
      if (layoutIndex == addIndex) {
        QHBoxLayout* newEntry = createLUTEntry(_colorLookupTables[_currentLUT.toStdString()], addIndex + 1);
        qobject_cast<QDoubleSpinBox*>(newEntry->itemAt(1)->widget())->setValue(newValue);
        QPixmap square(25, 25);        
        square.fill(QColor(newEntryColor[0], newEntryColor[1], newEntryColor[2], newEntryColor[3]));
        qobject_cast<QToolButton*>(newEntry->itemAt(0)->widget())->setIcon(square);
        parentLayout->insertLayout(addIndex + 1, newEntry);
      }
    }
  }    
  parentLayout->update();

  this->updateObjectNames();
  if (_editingLUT && _previewingLUT) {
    onLUTEntryChanged();
  }
}

void VisualizationWorkstationExtensionPlugin::removeLUTEntry() {
  QToolButton* removeButton = qobject_cast<QToolButton*>(sender());
  QString removeName = removeButton->objectName();
  QStringList nameList = removeName.split("_");
  int removeIndex = nameList[1].toInt();
  _colorLookupTables[_currentLUT.toStdString()].colors.erase(_colorLookupTables[_currentLUT.toStdString()].colors.begin() + removeIndex);
  _colorLookupTables[_currentLUT.toStdString()].indices.erase(_colorLookupTables[_currentLUT.toStdString()].indices.begin() + removeIndex);

  QLayout* parentLayout = qobject_cast<QWidget*>(removeButton->parent())->layout();
  for (auto editEnty : parentLayout->children()) {
    QHBoxLayout* hLayout = qobject_cast<QHBoxLayout*>(editEnty);
    if (hLayout) {
      QString hlayoutName = hLayout->objectName();
      QStringList nameAndIndex = hlayoutName.split("_");
      int layoutIndex = nameAndIndex[1].toInt();
      if (layoutIndex == removeIndex) {
        while (auto child = hLayout->takeAt(0)) {
          if (child->widget()) {
            child->widget()->deleteLater();
          }
        }
        delete hLayout;
      }
    }
  }
  parentLayout->update();
  this->updateObjectNames();
  if (_editingLUT && _previewingLUT) {
    onLUTEntryChanged();
  }
}

void VisualizationWorkstationExtensionPlugin::resetAllLUTs() {
  QMessageBox::StandardButton reply = QMessageBox::question(_LUTEditor, "Reset all LUTs", "Are you sure you want to reset all LUTs to the default? You might lose custom LUTs.", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    _colorLookupTables = pathology::DefaultColorLookupTables;
    QComboBox* LUTBox = _dockWidget->findChild<QComboBox*>("LUTComboBox");
    QComboBox* LUTEditorBox = _LUTEditor->findChild<QComboBox*>("LUTListComboBox");
    LUTBox->blockSignals(true);
    LUTBox->clear();
    LUTEditorBox->blockSignals(true);
    LUTEditorBox->clear();
    for (std::map<std::string, pathology::LUT>::const_iterator it = _colorLookupTables.begin(); it != _colorLookupTables.end(); ++it) {
      LUTBox->addItem(QString::fromStdString(it->first));
      LUTEditorBox->addItem(QString::fromStdString(it->first));
    }
    LUTBox->blockSignals(false);
    LUTEditorBox->blockSignals(false);
    _editingLUT = false;
    onLUTChanged(QString::fromStdString(_colorLookupTables.begin()->first));
    _editingLUT = true;
  }
}

void VisualizationWorkstationExtensionPlugin::addLUT() {
  bool ok;
  QString text = QInputDialog::getText(_LUTEditor, "Enter LUT name", "Provide a name for the new LUT:", QLineEdit::EchoMode::Normal, "", &ok);
  if (ok && !text.isEmpty()) {
    if (_colorLookupTables.find(text.toStdString()) == _colorLookupTables.end()) {
      std::vector<float> indices = { 0., 1. };
      std::vector<rgbaArray> colors = { {0,0,0,255}, {255,255,255,255} };
      pathology::LUT newLUT = { indices, colors };
      _colorLookupTables[text.toStdString()] = newLUT;
      QComboBox* LUTBox = _dockWidget->findChild<QComboBox*>("LUTComboBox");
      QComboBox* LUTEditorBox = _LUTEditor->findChild<QComboBox*>("LUTListComboBox");
      LUTBox->addItem(text);
      LUTEditorBox->addItem(text);
      onLUTChanged(text);
    }
    else {
      QMessageBox::warning(_LUTEditor, "LUT not created!", "The name you specified already exists, LUT was not created.", QMessageBox::Ok, QMessageBox::Ok);
    }
  }
}

void VisualizationWorkstationExtensionPlugin::removeLUT() {
  QMessageBox::StandardButton reply = QMessageBox::question(_LUTEditor, "Remove LUT", "Are you sure you want to remove this LUT? This cannot be undone.", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    if (_colorLookupTables.size() > 1) {
      std::string LUTToRemove = _currentLUT.toStdString();
      QComboBox* LUTBox = _dockWidget->findChild<QComboBox*>("LUTComboBox");
      QComboBox* LUTEditorBox = _LUTEditor->findChild<QComboBox*>("LUTListComboBox");
      LUTBox->blockSignals(true);
      LUTBox->removeItem(LUTBox->currentIndex());
      LUTEditorBox->blockSignals(true);
      LUTEditorBox->removeItem(LUTEditorBox->currentIndex());
      LUTBox->blockSignals(false);
      LUTEditorBox->blockSignals(false);
      _editingLUT = false;
      _colorLookupTables.erase(LUTToRemove);
      onLUTChanged(QString::fromStdString(_colorLookupTables.begin()->first));
      _editingLUT = true;
    }
    else {
      QMessageBox::warning(_LUTEditor, "LUT not removed!", "You cannot remove the last LUT from ASAP.", QMessageBox::Ok, QMessageBox::Ok);
    }
  }
}

void VisualizationWorkstationExtensionPlugin::duplicateLUT() {
  bool ok;
  QString text = QInputDialog::getText(_LUTEditor, "Enter LUT name", "Provide a name for the new LUT:", QLineEdit::EchoMode::Normal, _currentLUT + QString("_duplicate"), &ok);
  if (ok && !text.isEmpty()) {
    if (_colorLookupTables.find(text.toStdString()) == _colorLookupTables.end()) {
      _colorLookupTables[text.toStdString()] = _colorLookupTables[_currentLUT.toStdString()];
      QComboBox* LUTBox = _dockWidget->findChild<QComboBox*>("LUTComboBox");
      QComboBox* LUTEditorBox = _LUTEditor->findChild<QComboBox*>("LUTListComboBox");
      LUTBox->addItem(text);
      LUTEditorBox->addItem(text);
      onLUTChanged(text);
    }
    else {
      QMessageBox::warning(_LUTEditor, "LUT not created!",   "The name you specified already exists, LUT was not created.",  QMessageBox::Ok, QMessageBox::Ok);
    }
  }
}

void VisualizationWorkstationExtensionPlugin::handleEditLUTRequest()
{
  if (_LUTEditor) {
    _editingLUT = true;
    _LUTsBeforeEdit = _colorLookupTables;
    _currentLUTBeforeEdit = _currentLUT;
    int result = _LUTEditor->exec();
    _editingLUT = false;
    if (result) {
      onLUTChanged(_currentLUT);
    }
    else {
      _colorLookupTables = _LUTsBeforeEdit;
      onLUTChanged(_currentLUTBeforeEdit);
      _currentLUTBeforeEdit = "";
      _LUTsBeforeEdit.clear();
    }
  }
}

void VisualizationWorkstationExtensionPlugin::updateLUTPreviewStatus(int newCheckedState) {
  _previewingLUT = (newCheckedState == Qt::Checked);
  if (_previewingLUT) {
    onLUTChanged(_currentLUT);
  }
}

void VisualizationWorkstationExtensionPlugin::updateObjectNames() {
  if (this->_LUTEditingArea) {
    int objectIndex = 0;
    for (int i = 0; i < this->_LUTEditingArea->layout()->count(); ++i) {
      if (QHBoxLayout* hLayout = qobject_cast<QHBoxLayout*>(this->_LUTEditingArea->layout()->itemAt(i)->layout())) {
        hLayout->setObjectName(QString("editingEntry_") + QString::number(objectIndex));
        for (unsigned int i = 0; i < hLayout->count(); ++i) {
          QWidget* entryWidget = hLayout->itemAt(i)->widget();
          QString entryWidgetName = entryWidget->objectName();
          QString nameWithoutIndex = entryWidgetName.split("_")[0];
          entryWidget->setObjectName(nameWithoutIndex + QString("_") + QString::number(objectIndex));
        }
        objectIndex++;
      }
    }
  }
}

void VisualizationWorkstationExtensionPlugin::clearForegroundImage()
{
  if (_foreground) {
    _foregroundScale = 1;
    emit changeForegroundImage(std::weak_ptr<MultiResolutionImage>(), _foregroundScale);
    _foreground.reset();
  }
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
    this->loadNewForegroundImage(likImgPth);
  }
}

void VisualizationWorkstationExtensionPlugin::onOpenResultImageClicked() {
  QString fileName = QFileDialog::getOpenFileName(_dockWidget, tr("Open File"), _settings->value("lastOpenendPath", QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)).toString(), tr("Result files (*.tif);;All files (*.*)"));
  if (!fileName.isEmpty()) {
    loadNewForegroundImage(fileName.toStdString());
  }
}

void VisualizationWorkstationExtensionPlugin::loadNewForegroundImage(const std::string& resultImagePth) {
  this->clearForegroundImage();
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
        if (_likelihoodCheckBox) {
          if (_renderingEnabled) {
            if (_likelihoodCheckBox->isChecked()) {
              emit changeForegroundImage(_foreground, _foregroundScale);
            }
            else {
              _likelihoodCheckBox->setChecked(true);
            }
          }
          else {
            if (_likelihoodCheckBox->isChecked()) {
              _likelihoodCheckBox->setChecked(false);
            }
            else {
              emit changeForegroundImage(std::weak_ptr<MultiResolutionImage>(), _foregroundScale);
            }
          }
        }
      }
      QGroupBox* visualizationGroupBox = _dockWidget->findChild<QGroupBox*>("VisualizationGroupBox");
      visualizationGroupBox->setEnabled(true);
    }
  }
}

void VisualizationWorkstationExtensionPlugin::setDefaultVisualizationParameters(std::shared_ptr<MultiResolutionImage> img) {
  if (_dockWidget) {
    if (_settings) {
      _settings->beginGroup("VisualizationWorkstationExtensionPlugin");
      pathology::DataType dtype = img->getDataType();
      if (dtype == pathology::Float) {
        _settings->beginGroup("VisualizationSettingsForFloatType");
      }
      else if (dtype == pathology::UChar) {
        _settings->beginGroup("VisualizationSettingsForUCharType");
      }
      else if (dtype == pathology::UInt16) {
        _settings->beginGroup("VisualizationSettingsForUInt16Type");
      }
      else if (dtype == pathology::UInt32) {
        _settings->beginGroup("VisualizationSettingsForUInt32Type");
      }
      _opacity = _settings->value("opacity", 0.5).toFloat();
      _foregroundChannel = _settings->value("foregroundchannel", 0).toUInt();
      if (_foregroundChannel >= img->getSamplesPerPixel()) {
        _foregroundChannel = 0;
      }
      if (dtype == pathology::Float) {
        _currentLUT = _settings->value("lut", "Traffic Light (0 - 1)").toString();
      }
      else {
        _currentLUT = _settings->value("lut", "Label").toString();
      }
      _renderingEnabled = _settings->value("visible", false).toBool();
      _settings->endGroup();
      _settings->endGroup();
    }
    else {
      _opacity = 0.5;
      _foregroundChannel = 0;
      if (img->getDataType() == pathology::UChar || img->getDataType() == pathology::UInt32 || img->getDataType() == pathology::UInt16) {
        _currentLUT = "Label";
      }
      else {
        _currentLUT = "Traffic Light (0 - 1)";
      }
    }
    QDoubleSpinBox* spinBox = _dockWidget->findChild<QDoubleSpinBox*>("OpacitySpinBox");
    spinBox->setValue(_opacity);
    _viewer->setForegroundOpacity(_opacity);
    QSpinBox* channelSpinBox = _dockWidget->findChild<QSpinBox*>("ChannelSpinBox");
    channelSpinBox->setMaximum(_foreground->getSamplesPerPixel() - 1);
    channelSpinBox->setValue(_foregroundChannel);
    _viewer->setForegroundChannel(_foregroundChannel);
    onLUTChanged(_currentLUT);
  }
}

void VisualizationWorkstationExtensionPlugin::onImageClosed() {
  // Store current visualization settings based on ImageType (later replace this with Result specific settings)
  if (_settings && _foreground) {
    _settings->beginGroup("VisualizationWorkstationExtensionPlugin");
    pathology::DataType dtype = _foreground->getDataType();
    if (dtype == pathology::Float) {
      _settings->beginGroup("VisualizationSettingsForFloatType");
    }
    else if (dtype == pathology::UChar) {
      _settings->beginGroup("VisualizationSettingsForUCharType");
    }
    else if (dtype == pathology::UInt16) {
      _settings->beginGroup("VisualizationSettingsForUInt16Type");
    }
    else if (dtype == pathology::UInt32) {
      _settings->beginGroup("VisualizationSettingsForUInt32Type");
    }
    _settings->setValue("opacity", _opacity);
    _settings->setValue("foregroundchannel", _foregroundChannel);
    _settings->setValue("lut", _currentLUT);
    _settings->setValue("visible", _renderingEnabled);
    _settings->endGroup();
    _settings->endGroup();
  }
  this->clearForegroundImage();
  if (_dockWidget) {
    _dockWidget->setEnabled(false);
    QGroupBox* visualizationGroupBox = _dockWidget->findChild<QGroupBox*>("VisualizationGroupBox");
    visualizationGroupBox->setEnabled(false);
  }
}

// FIX: Just disable rendering
void VisualizationWorkstationExtensionPlugin::onEnableLikelihoodToggled(bool toggled) {
  if (!toggled) {
    emit changeForegroundImage(std::weak_ptr<MultiResolutionImage>(), _foregroundScale);
    _renderingEnabled = false;
  }
  else {
    emit changeForegroundImage(_foreground, _foregroundScale);
    _renderingEnabled = true;
  }
}

void VisualizationWorkstationExtensionPlugin::onOpacityChanged(double opacity) {
  if (_viewer) {
    _viewer->setForegroundOpacity(opacity);
    _opacity = opacity;
  }
}

void VisualizationWorkstationExtensionPlugin::onLUTChanged(const QString& LUTname) {
  if (_viewer) {
    _currentLUT = LUTname;
    QComboBox* LUTBox = _dockWidget->findChild<QComboBox*>("LUTComboBox");
    LUTBox->setCurrentText(LUTname);
    QComboBox* LUTEditorBox = _LUTEditor->findChild<QComboBox*>("LUTListComboBox");
    LUTEditorBox->setCurrentText(LUTname);
    this->generateLUTEditingWidgets(_currentLUT);
    if (_editingLUT && _previewingLUT) {
      _viewer->setForegroundLUT(_colorLookupTables[_currentLUT.toStdString()]);
    }
    else if (!_editingLUT) {
      _viewer->setForegroundLUT(_colorLookupTables[_currentLUT.toStdString()]);
    }
  }
}

void VisualizationWorkstationExtensionPlugin::onLUTEntryChanged() {
  if (_viewer) {
    if (_editingLUT && _previewingLUT) {
      _viewer->setForegroundLUT(_colorLookupTables[_currentLUT.toStdString()]);
    }
    else if (!_editingLUT) {
      _viewer->setForegroundLUT(_colorLookupTables[_currentLUT.toStdString()]);
    }
  }
}

void VisualizationWorkstationExtensionPlugin::onChannelChanged(int channel) {
  if (_viewer && channel != _foregroundChannel) {
    _foregroundChannel = channel;
    _viewer->setForegroundChannel(_foregroundChannel);
  }
}