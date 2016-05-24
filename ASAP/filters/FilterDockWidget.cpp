#include "FilterDockWidget.h"

#include <QListWidget>
#include <QLayout>
#include <QLabel>
#include <QProgressBar>
#include <QCheckBox>
#include <QPushButton>
#include <QtUiTools>

#include "interfaces/interfaces.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "pathologyworkstation.h"
#include "QtProgressMonitor.h"

FilterDockWidget::FilterDockWidget(QWidget *parent, Qt::WindowFlags flags) : 
  QDockWidget("Image Filters", parent, flags),
  _currentSelection(NULL),
  _availableFilters(NULL),
  _settingsPanel(NULL),
  _layout(NULL),
  _progressBar(NULL),
  _monitor(NULL),
  _applyFilter(NULL),
  _clearFilter(NULL),
  _autoUpdateCheckBox(NULL),
  _autoUpdate(false)
{
  QUiLoader loader;
  QFile file(":/FilterWorkstationExtensionPlugin_ui/FilterDockWidget.ui");
  file.open(QFile::ReadOnly | QFile::Text);
  QWidget* contents = loader.load(&file);
  file.close();
  _layout = contents->layout();
  _layout->setAlignment(Qt::AlignTop);
  _availableFilters = contents->findChild<QListWidget*>("filterListWidget");
  connect(_availableFilters, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(onItemClicked(QListWidgetItem*)));
  _progressBar = contents->findChild<QProgressBar*>("progressBar");
  QObject::connect(_progressBar, SIGNAL(valueChanged(int)), this, SLOT(onProcessing()));
  _monitor.reset(new QtProgressMonitor());
  QObject::connect(_monitor.get(), SIGNAL(progressChanged(int)), _progressBar, SLOT(setValue(int)), Qt::QueuedConnection);
  _applyFilter = contents->findChild<QPushButton*>("applyFilterButton");
  _clearFilter = contents->findChild<QPushButton*>("clearFilterButton");
  _autoUpdateCheckBox = contents->findChild<QCheckBox*>("autoUpdateCheckBox");
  QObject::connect(_autoUpdateCheckBox, SIGNAL(toggled(bool)), _applyFilter, SLOT(setDisabled(bool)));
  QObject::connect(_autoUpdateCheckBox, SIGNAL(toggled(bool)), this, SLOT(onAutoUpdateToggled(bool)));
  QObject::connect(_applyFilter, SIGNAL(clicked()), this, SLOT(onApplyFilterClicked()));
  QObject::connect(_clearFilter, SIGNAL(clicked()), this, SLOT(onClearFilterClicked()));
  this->setWidget(contents);
}

FilterDockWidget::~FilterDockWidget() {
  stopProgressTracking();
}

void FilterDockWidget::onAutoUpdateToggled(bool autoUpdate) {
  _autoUpdate = autoUpdate;
  emit changeAutoUpdateStatus(autoUpdate);
}

void FilterDockWidget::onApplyFilterClicked() {
  emit requestFilterResultUpdate();
}

void FilterDockWidget::onClearFilterClicked() {
  emit requestFilterResultClear();
}

void FilterDockWidget::onFilterParametersChanged() {
  if (_autoUpdate) {
    emit requestFilterResultUpdate();
  }
}

void FilterDockWidget::onProcessing() {
  if (_progressBar) {
    if (_progressBar->value() == _progressBar->maximum()) {
      _progressBar->setValue(_progressBar->minimum());
    }
  }
}

void FilterDockWidget::onItemClicked(QListWidgetItem* clickedItem) {
  if (_availableFilters) {
    std::shared_ptr<ImageFilterPluginInterface> filter = NULL;
    if (_settingsPanel) {
      _layout->removeWidget(_settingsPanel);
      _settingsPanel->hide();
      _settingsPanel->setParent(NULL);
      _settingsPanel = NULL;
    }
    stopProgressTracking();
    if (clickedItem && clickedItem != _currentSelection) {
      if (_currentSelection) {
        std::shared_ptr<ImageFilterPluginInterface> filter = _currentSelection->data(Qt::UserRole).value<std::shared_ptr<ImageFilterPluginInterface> >();
        filter->setProgressMonitor(NULL);
        filter->disconnect(SIGNAL(filterParametersChanged()));
      }
      _currentSelection = clickedItem;
      filter = _currentSelection->data(Qt::UserRole).value<std::shared_ptr<ImageFilterPluginInterface> >();
      filter->setProgressMonitor(_monitor);
      _settingsPanel = filter->getSettingsPanel();        
      _layout->addWidget(_settingsPanel);
      QObject::connect(filter.get(), SIGNAL(filterParametersChanged()), this, SLOT(onFilterParametersChanged()));
    }
    else if (clickedItem && clickedItem == _currentSelection) {
      _availableFilters->clearSelection();
      std::shared_ptr<ImageFilterPluginInterface> filter = _currentSelection->data(Qt::UserRole).value<std::shared_ptr<ImageFilterPluginInterface> >();
      filter->setProgressMonitor(NULL);
      filter->disconnect(SIGNAL(filterParametersChanged()));
      _currentSelection = NULL;
    }
    if (_currentSelection) {
      _autoUpdateCheckBox->setEnabled(true);
      if (_autoUpdateCheckBox->isChecked()) {
        _applyFilter->setEnabled(false);
      }
      else {
        _applyFilter->setEnabled(true);
      }
    }
    else {
      _applyFilter->setEnabled(false);
      _autoUpdateCheckBox->setEnabled(false);
    }
    emit changeCurrentFilter(filter);
  }
}

void FilterDockWidget::stopProgressTracking() {
  if (_currentSelection) {
    std::shared_ptr<ImageFilterPluginInterface> filter = _currentSelection->data(Qt::UserRole).value<std::shared_ptr<ImageFilterPluginInterface> >();
    if (filter) {
      filter->setProgressMonitor(NULL);
    }
    _progressBar->setValue(0);
  }
}

void FilterDockWidget::onNewImageLoaded(std::weak_ptr<MultiResolutionImage> img) {
  if (_availableFilters) {
    for (int row = 0; row < _availableFilters->count(); row++)
    {
      QListWidgetItem *item = _availableFilters->item(row);
      std::shared_ptr<ImageFilterPluginInterface> filter = item->data(Qt::UserRole).value<std::shared_ptr<ImageFilterPluginInterface> >();
      std::shared_ptr<MultiResolutionImage> local_img = img.lock();
      if (!filter->initialize(local_img.get())) {
        item->setHidden(true);
      }
      else {
        item->setHidden(false);
      }
    }
  }
}

void FilterDockWidget::onImageClosed() {
  stopProgressTracking();
  if (_settingsPanel) {
    _layout->removeWidget(_settingsPanel);
    _settingsPanel->hide();
    _settingsPanel->setParent(NULL);
    _settingsPanel = NULL;
  }
  if (_availableFilters) {
    _availableFilters->clearSelection();
    _currentSelection = NULL;
    for (int row = 0; row < _availableFilters->count(); row++)
    {
      QListWidgetItem *item = _availableFilters->item(row);
      std::shared_ptr<ImageFilterPluginInterface> filter = item->data(Qt::UserRole).value<std::shared_ptr<ImageFilterPluginInterface> >();
      item->setHidden(true);
    }
  }
}