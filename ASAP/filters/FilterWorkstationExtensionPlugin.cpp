#include "FilterWorkstationExtensionPlugin.h"
#include "FilterDockWidget.h"
#include "FilterThread.h"
#include "PathologyViewController.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"

#include <QDockWidget>
#include <QDir>
#include <QApplication>
#include <QPluginLoader>
#include <QListWidgetItem>
#include <QGraphicsItem>

FilterWorkstationExtensionPlugin::FilterWorkstationExtensionPlugin() :
  WorkstationExtensionPluginInterface(),
  _dockWidget(NULL),
  _filterThread(NULL),
  _filterResult(NULL),
  _autoUpdate(false)
{
  _dockWidget = new FilterDockWidget();
  _dockWidget->setEnabled(false);
  _dockWidget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  connect(_dockWidget, SIGNAL(requestFilterResultUpdate()), this, SLOT(onFilterResultUpdateRequested()));
  connect(_dockWidget, SIGNAL(requestFilterResultClear()), this, SLOT(onFilterResultClearRequested()));
  connect(_dockWidget, SIGNAL(changeAutoUpdateStatus(bool)), this, SLOT(onAutoUpdateStatusChanged(bool)));
  _filterThread = new FilterThread();
  connect(_filterThread, SIGNAL(filterResult(QGraphicsItem*, QRectF)), this, SLOT(updateFilteredImage(QGraphicsItem*, QRectF)));
  QDir pluginsDir = QDir(qApp->applicationDirPath());
  pluginsDir.cd("plugins");
  if (pluginsDir.cd("filters")) {
    QListWidget* availableFilters = _dockWidget->findChild<QListWidget*>("filterListWidget");
    foreach(QString fileName, pluginsDir.entryList(QDir::Files)) {
#ifdef WIN32
      if (fileName.toLower().endsWith(".dll")) {
#else
      if (fileName.toLower().endsWith(".so")) {
#endif
        QPluginLoader loader(pluginsDir.absoluteFilePath(fileName));
        QObject *plugin = loader.instance();
        if (plugin) {
          std::shared_ptr<ImageFilterPluginInterface> filter(qobject_cast<ImageFilterPluginInterface *>(plugin));
          if (filter) {
            if (_dockWidget) {
              QListWidgetItem* filterItem = new QListWidgetItem(filter->icon(), filter->name());
              filterItem->setData(Qt::ItemDataRole::UserRole, QVariant::fromValue(filter));
              availableFilters->addItem(filterItem);
              filterItem->setHidden(true);
            }
          }
        }
      }
    }
  }
}

FilterWorkstationExtensionPlugin::~FilterWorkstationExtensionPlugin() {
  _dockWidget = NULL;
  if (_filterThread) {
    delete _filterThread;
    _filterThread = NULL;
  }
}

void FilterWorkstationExtensionPlugin::onFilterResultClearRequested() {
  if (_filterThread) {
    _filterThread->stopFilter();
  }
  if (_filterResult) {
    _filterResult->setVisible(false);
    _controller->GetMasterViewer()->scene()->removeItem(_filterResult);
    delete _filterResult;
    _filterResult = NULL;
  }
}

void FilterWorkstationExtensionPlugin::onFilterResultUpdateRequested() {
  PathologyViewer* viewer(_controller->GetMasterViewer());
  if (_filterThread) {
    if (std::shared_ptr<MultiResolutionImage> local_img = viewer->GetDocumentInstance()->document->GetImage().lock()) {
      float sceneScale = viewer->getSceneScale();
      float maxDownsample = 1. / sceneScale;
      QRectF FOV = viewer->mapToScene(viewer->rect()).boundingRect();
      QRectF FOVImage = QRectF(FOV.left() / sceneScale, FOV.top() / sceneScale, FOV.width() / sceneScale, FOV.height() / sceneScale);
      _filterThread->updateFilterResult(FOVImage, viewer->GetDocumentInstance()->document->GetImage(), local_img->getBestLevelForDownSample(maxDownsample / viewer->transform().m11()), -1);
    }
  }
}

void FilterWorkstationExtensionPlugin::updateFilteredImage(QGraphicsItem* result, QRectF size) {
  PathologyViewer* viewer(_controller->GetMasterViewer());
  onFilterResultClearRequested();
  if (result) {
    result->setVisible(false);
    _filterResult = result;
	viewer->scene()->addItem(_filterResult);
    _filterResult->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
    _filterResult->setPos(viewer->mapToScene(viewer->rect()).boundingRect().topLeft());
    _filterResult->setTransform(QTransform::fromScale(static_cast<float>(viewer->width()) / size.width(), static_cast<float>(viewer->height()) / size.height()), true);
    _filterResult->setVisible(true);
    _filterResult->setZValue(10.);
  }
}

void FilterWorkstationExtensionPlugin::onChangeCurrentFilter(std::shared_ptr<ImageFilterPluginInterface> filter) {
  onFilterResultClearRequested();
  if (_filterThread && !filter) {
    _filterThread->removeFilter();
  }
  else {
    _filterThread->setFilter(filter);
  }
  if (_autoUpdate) {
    onFilterResultUpdateRequested();
  }
}

void FilterWorkstationExtensionPlugin::onAutoUpdateStatusChanged(bool autoUpdate) {
  _autoUpdate = autoUpdate;
  if (_filterThread) {
    onFilterResultUpdateRequested();
  }
}

QDockWidget* FilterWorkstationExtensionPlugin::getDockWidget() {
  return _dockWidget;
}

void FilterWorkstationExtensionPlugin::onFieldOfViewChanged(const QRectF& FOV, const unsigned int level) {
  onFilterResultClearRequested();
  if (_filterThread && _autoUpdate) {
    onFilterResultUpdateRequested();
  }
}

void FilterWorkstationExtensionPlugin::onDocumentChange(ASAP::DocumentInstance* document)
{
	if (document)
	{
		if (_dockWidget) {
			_dockWidget->setEnabled(true);
			_dockWidget->onNewImageLoaded(document->document->GetImage());

			connect(_dockWidget,
				&FilterDockWidget::changeCurrentFilter,
				this,
				&FilterWorkstationExtensionPlugin::onChangeCurrentFilter);
		}
	}
	else
	{
		if (_filterResult) {
			onFilterResultClearRequested();
		}
		if (_dockWidget) {
			_dockWidget->setEnabled(false);
			_dockWidget->onImageClosed();

			disconnect(_dockWidget,
				&FilterDockWidget::changeCurrentFilter,
				this,
				&FilterWorkstationExtensionPlugin::onChangeCurrentFilter);
		}
	}
}

void FilterWorkstationExtensionPlugin::prepareForViewerChange_(void)
{
	PathologyViewer* viewer(_controller->GetMasterViewer());
	if (viewer)
	{
		disconnect(viewer,
			&PathologyViewer::fieldOfViewChanged,
			this,
			&FilterWorkstationExtensionPlugin::onFieldOfViewChanged);
	}
}

void FilterWorkstationExtensionPlugin::setupNewViewer_(void)
{
	PathologyViewer* viewer(_controller->GetMasterViewer());
	if (viewer)
	{
		connect(viewer,
			&PathologyViewer::fieldOfViewChanged,
			this,
			&FilterWorkstationExtensionPlugin::onFieldOfViewChanged);
	}
}