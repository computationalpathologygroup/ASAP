#include "PathologyViewer.h"

#include <iostream>

#include <QResizeEvent>
#include <QApplication>
#include <QMenu>
#include <QAction>
#include <QMessageBox>
#include <QGLWidget>
#include <QTimeLine>
#include <QScrollBar>
#include <QHBoxLayout>
#include <QSettings>
#include <QGuiApplication>
#include <QMainWindow>
#include <QStatusBar>

#include "MiniMap.h"
#include "ScaleBar.h"
#include "RenderThread.h"
#include "PrefetchThread.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "interfaces/interfaces.h"
#include "core/PathologyEnums.h"
#include "WSITileGraphicsItem.h"
#include "WSITileGraphicsItemCache.h"
#include "TileManager.h"
#include "RenderWorker.h"

using std::vector;

PathologyViewer::PathologyViewer(WSITileGraphicsItemCache& cache, QWidget* parent) :
	QGraphicsView(parent),
	_sceneScale(1.0f),
	_renderthread(nullptr),
	_prefetchthread(nullptr),
	_map(nullptr),
	_cache(cache),
	_manager(nullptr),
	_scaleBar(nullptr),
	m_instance_(nullptr)
{
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setResizeAnchor(QGraphicsView::ViewportAnchor::AnchorViewCenter);
  setDragMode(QGraphicsView::DragMode::NoDrag);
  setContentsMargins(0,0,0,0);
  setAutoFillBackground(true);
  setViewportUpdateMode(ViewportUpdateMode::FullViewportUpdate);
  setInteractive(false);
  this->setScene(new QGraphicsScene(this)); //Memleak!
  this->setBackgroundBrush(QBrush(QColor(252, 252, 252)));
  this->scene()->setBackgroundBrush(QBrush(QColor(252, 252, 252)));
  this->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(this, SIGNAL(customContextMenuRequested(const QPoint&)),
          this, SLOT(showContextMenu(const QPoint&)));
  _settings = new QSettings(QSettings::IniFormat, QSettings::UserScope, "DIAG", "ASAP", this);
  _settings->beginGroup("ASAP");
  if (this->window()) {
    QMenu* viewMenu = this->window()->findChild<QMenu*>("menuView");
    QAction* action;
    if (viewMenu) {
      action = viewMenu->addAction("Toggle scale bar");
      action->setCheckable(true);
      action->setChecked(_settings->value("scaleBarToggled", true).toBool());
      action = viewMenu->addAction("Toggle coverage view");
      action->setCheckable(true);
      action->setChecked(_settings->value("coverageViewToggled", true).toBool());
      action = viewMenu->addAction("Toggle mini-map");
      action->setCheckable(true);
      action->setChecked(_settings->value("miniMapToggled", true).toBool());
    }
  }
  _settings->endGroup();
}

PathologyViewer::~PathologyViewer()
{
  close();
}

void PathologyViewer::resizeEvent(QResizeEvent *event) {
	QGraphicsView::resizeEvent(event);
	if (m_instance_)
	{
		this->refreshView();
	}
}

float PathologyViewer::getSceneScale(void) const
{
	return _sceneScale;
}

ASAP::DocumentInstance* PathologyViewer::GetDocumentInstance(void)
{
	return m_instance_;
}

void PathologyViewer::moveTo(const QPointF& pos) {
  this->centerOn(pos);
  this->refreshView();
}

void PathologyViewer::onFieldOfViewChanged(const QRectF& FOV, const unsigned int level) {
  if (_manager) {
    _manager->loadTilesForFieldOfView(FOV, level);
  }
}

void PathologyViewer::initialize(ASAP::DocumentInstance& instance) {
  // Closes the current image and inserts a new instance.
  close();
  setEnabled(true);
  m_instance_ = &instance;

  // Setups internals
  _renderthread = new RenderThread(this);
  _renderthread->setBackgroundImage(m_instance_->document->GetImage());
  _manager = new TileManager(*m_instance_, _renderthread, &_cache, scene());
  setMouseTracking(true);
  std::vector<RenderWorker*> workers = _renderthread->getWorkers();
  for (int i = 0; i < workers.size(); ++i) {
    QObject::connect(workers[i], SIGNAL(tileLoaded(QPixmap*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)), _manager, SLOT(onTileLoaded(QPixmap*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)));
  }

  // Initializes GUI elements
  ASAP::TileInformation tile_info = m_instance_->document->GetTileInformation();

  initializeImage(tile_info.tile_size, tile_info.top_level);
  initializeGUIComponents(tile_info.top_level);
  QObject::connect(this, SIGNAL(backgroundChannelChanged(int)), _renderthread, SLOT(onBackgroundChannelChanged(int)));
  QObject::connect(&_cache, SIGNAL(itemEvicted(WSITileGraphicsItem*)), _manager, SLOT(onTileRemoved(WSITileGraphicsItem*)));
  QObject::connect(this, SIGNAL(fieldOfViewChanged(const QRectF, const unsigned int)), this, SLOT(onFieldOfViewChanged(const QRectF, const unsigned int)));

  this->setViewState(m_instance_->view_state);
}

void PathologyViewer::onForegroundImageChanged(std::weak_ptr<MultiResolutionImage> for_img, float scale) {
  if (m_instance_ && _renderthread) {
	  m_instance_->view_state.foreground_image = for_img;
	  m_instance_->view_state.foreground_scale = scale;

    _renderthread->setForegroundImage(m_instance_->view_state.foreground_image, scale);
    _manager->refresh();
  }
}

void PathologyViewer::setForegroundLUT(const std::string& LUTname) {
  if (m_instance_ && _renderthread) {
	  m_instance_->view_state.foreground_lut_name = LUTname;

    _renderthread->onLUTChanged(LUTname);
    if (m_instance_->view_state.foreground_image.lock()) {
      _manager->refresh();
    }
  }
}

void PathologyViewer::setForegroundWindowAndLevel(const float& window, const float& level) {
  if (m_instance_ && _renderthread) {
	  m_instance_->view_state.foreground_window = window;
	  m_instance_->view_state.foreground_level	= level;

    _renderthread->onWindowAndLevelChanged(window, level);
    if (m_instance_->view_state.foreground_image.lock()) {
      _manager->refresh();
    }
  }
}

void PathologyViewer::setForegroundChannel(unsigned int channel) {
  if (m_instance_ && _renderthread) {
	  m_instance_->view_state.foreground_channel = channel;
    _renderthread->onForegroundChannelChanged(channel);
    if (m_instance_->view_state.foreground_image.lock()) {
      _manager->refresh();
    }
  }
}


void PathologyViewer::setForegroundOpacity(const float& opacity) {
  if (_renderthread) {
    _renderthread->setForegroundOpacity(opacity);
    if (m_instance_->view_state.foreground_image.lock()) {
      _manager->refresh();
    }
  }
}


float PathologyViewer::getForegroundOpacity() const {
  return m_instance_->view_state.foreground_opacity;
}

void PathologyViewer::initializeImage(unsigned int tileSize, unsigned int lastLevel) {  
  std::vector<unsigned long long> lastLevelDimensions = m_instance_->document->image().getLevelDimensions(lastLevel);
  float lastLevelWidth = ((lastLevelDimensions[0] / tileSize) + 1) * tileSize;
  float lastLevelHeight = ((lastLevelDimensions[1] / tileSize) + 1) * tileSize;
  float longest = lastLevelWidth > lastLevelHeight ? lastLevelWidth : lastLevelHeight;
  _sceneScale = 1. / m_instance_->document->image().getLevelDownsample(lastLevel);
  QRectF n((lastLevelDimensions[0] / 2) - 1.5*longest, (lastLevelDimensions[1] / 2) - 1.5*longest, 3 * longest, 3 * longest);
  this->setSceneRect(n);
  this->fitInView(QRectF(0, 0, lastLevelDimensions[0], lastLevelDimensions[1]), Qt::AspectRatioMode::KeepAspectRatio);

  _manager->loadAllTilesForLevel(lastLevel);
  this->refreshView();
  while (_renderthread->numberOfJobs() > 0) {
  }
}

void PathologyViewer::initializeGUIComponents(unsigned int level) {
  // Initialize the minimap
  std::vector<unsigned long long> overviewDimensions = m_instance_->document->image().getLevelDimensions(level);
  unsigned int size = overviewDimensions[0] * overviewDimensions[1] * m_instance_->document->image().getSamplesPerPixel();
  unsigned char* overview = new unsigned char[size];
  m_instance_->document->image().getRawRegion<unsigned char>(0, 0, overviewDimensions[0], overviewDimensions[1], level, overview);
  QImage ovImg;
  if (m_instance_->document->image().getColorType() == pathology::ARGB) {
    ovImg = QImage(overview, overviewDimensions[0], overviewDimensions[1], overviewDimensions[0] * 4, QImage::Format_ARGB32).convertToFormat(QImage::Format_RGB888);
  }
  else if (m_instance_->document->image().getColorType() == pathology::RGB) {
    ovImg = QImage(overview, overviewDimensions[0], overviewDimensions[1], overviewDimensions[0] * 3, QImage::Format_RGB888);
  }
  QPixmap ovPixMap = QPixmap(QPixmap::fromImage(ovImg));
  delete[] overview;
  if (_map) {
    _map->deleteLater();
    _map = NULL;
  }
  _map = new MiniMap(ovPixMap, this);
  if (_scaleBar) {
    _scaleBar->deleteLater();
    _scaleBar = NULL;
  }
  std::vector<double> spacing = m_instance_->document->image().getSpacing();
  if (!spacing.empty()) {
    _scaleBar = new ScaleBar(spacing[0], this);
  }
  else {
    _scaleBar = new ScaleBar(-1, this);
  }
  if (this->layout()) {
    delete this->layout();
  }
  QHBoxLayout * Hlayout = new QHBoxLayout(this);
  QVBoxLayout * Vlayout = new QVBoxLayout();
  QVBoxLayout * Vlayout2 = new QVBoxLayout();
  Vlayout2->addStretch(4);
  Hlayout->addLayout(Vlayout2);
  Hlayout->addStretch(4);
  Hlayout->setContentsMargins(30, 30, 30, 30);
  Hlayout->addLayout(Vlayout, 1);
  Vlayout->addStretch(4);
  if (_map) {
    Vlayout->addWidget(_map, 1);
  }
  if (_scaleBar) {
    Vlayout2->addWidget(_scaleBar);
  }
  _map->setTileManager(_manager);
  QObject::connect(this, SIGNAL(updateBBox(const QRectF&)), _map, SLOT(updateFieldOfView(const QRectF&)));
  QObject::connect(_manager, SIGNAL(coverageUpdated()), _map, SLOT(onCoverageUpdated()));
  QObject::connect(_map, SIGNAL(positionClicked(QPointF)), this, SLOT(moveTo(const QPointF&)));
  QObject::connect(this, SIGNAL(fieldOfViewChanged(const QRectF&, const unsigned int)), _scaleBar, SLOT(updateForFieldOfView(const QRectF&)));
  if (this->window()) {
    _settings->beginGroup("ASAP");
    QMenu* viewMenu = this->window()->findChild<QMenu*>("menuView");
    if (viewMenu)  {
      QList<QAction*> actions = viewMenu->actions();
      for (QList<QAction*>::iterator it = actions.begin(); it != actions.end(); ++it) {
        if ((*it)->text() == "Toggle scale bar" && _scaleBar) {
          QObject::connect((*it), SIGNAL(toggled(bool)), _scaleBar, SLOT(setVisible(bool)));
          bool showComponent = _settings->value("scaleBarToggled", true).toBool();
          (*it)->setChecked(showComponent);
          _scaleBar->setVisible(showComponent);
        }
        else if ((*it)->text() == "Toggle mini-map" && _map) {
          QObject::connect((*it), SIGNAL(toggled(bool)), _map, SLOT(setVisible(bool)));
          bool showComponent = _settings->value("miniMapToggled", true).toBool();
          (*it)->setChecked(showComponent);
          _map->setVisible(showComponent);
        }
        else if ((*it)->text() == "Toggle coverage view" && _map) {
          QObject::connect((*it), SIGNAL(toggled(bool)), _map, SLOT(toggleCoverageMap(bool)));
          bool showComponent = _settings->value("coverageViewToggled", true).toBool();
          (*it)->setChecked(showComponent);
          _map->toggleCoverageMap(showComponent);
        }
      }
    }
    _settings->endGroup();
  }
}

void PathologyViewer::showContextMenu(const QPoint& pos)
{
  QPoint globalPos = this->mapToGlobal(pos);

  if (m_instance_) {
    QMenu rightClickMenu;
    if (m_instance_->document->image().getColorType() == pathology::ColorType::Indexed) {
      for (int i = 0; i < m_instance_->document->image().getSamplesPerPixel(); ++i) {
        rightClickMenu.addAction(QString("Channel ") + QString::number(i+1));
      }
      QAction* selectedItem = rightClickMenu.exec(globalPos);
      if (selectedItem)
      {
        for (int i = 0; i < m_instance_->document->image().getSamplesPerPixel(); ++i) {
          if (selectedItem->text() == QString("Channel ") + QString::number(i + 1)) {
            emit backgroundChannelChanged(i);
            _manager->refresh();
          }
        }
      }
    }
    else if (m_instance_->document->image().getNumberOfZPlanes() > 1) {
      for (int i = 0; i < m_instance_->document->image().getNumberOfZPlanes(); ++i) {
        rightClickMenu.addAction(QString("Plane ") + QString::number(i + 1));
      }
      QAction* selectedItem = rightClickMenu.exec(globalPos);
      if (selectedItem)
      {
        for (int i = 0; i < m_instance_->document->image().getNumberOfZPlanes(); ++i) {
          if (selectedItem->text() == QString("Plane ") + QString::number(i + 1)) {
			  m_instance_->document->image().setCurrentZPlaneIndex(i);
            _manager->refresh();
          }
        }
      }
    }
  }
}

void PathologyViewer::close() {
  if (this->window())
  {
    QMenu* viewMenu = this->window()->findChild<QMenu*>("menuView");
    _settings->beginGroup("ASAP");
    if (viewMenu) {
      QList<QAction*> actions = viewMenu->actions();
      for (QList<QAction*>::iterator it = actions.begin(); it != actions.end(); ++it) {
        if ((*it)->text() == "Toggle scale bar" && _scaleBar) {
          _settings->setValue("scaleBarToggled", (*it)->isChecked());
        }
        else if ((*it)->text() == "Toggle mini-map" && _map) {
          _settings->setValue("miniMapToggled", (*it)->isChecked());
        }
        else if ((*it)->text() == "Toggle coverage view" && _map) {
          _settings->setValue("coverageViewToggled", (*it)->isChecked());
        }
      }
    }
    _settings->endGroup();
  }
  if (_prefetchthread) {
    _prefetchthread->deleteLater();
    _prefetchthread = NULL;
  }
  scene()->clear();
  if (_manager) {
    _manager->clear();
    delete _manager;
    _manager = NULL;
  }
  m_instance_ = NULL;
  if (_renderthread) {
    _renderthread->shutdown();
    _renderthread->deleteLater();
    _renderthread = NULL;
  }
  if (_map) {
    _map->setHidden(true);
    _map->deleteLater();
    _map = NULL;
  }
  if (_scaleBar) {
    _scaleBar->setHidden(true);
    _scaleBar->deleteLater();
    _scaleBar = NULL;
  }

  setEnabled(false);
  emit documentInstanceChanged(nullptr);
}

void PathologyViewer::togglePan(bool pan, const QPoint& startPos) {
	if (pan)
	{
		m_instance_->view_state.pan_position = startPos;
		setCursor(Qt::ClosedHandCursor);
	}
	else
	{
		m_instance_->view_state.pan_position = QPoint(0, 0);
		setCursor(Qt::ArrowCursor);
	}
}

void PathologyViewer::modifyPan(const QPoint& pan_to_point)
{
	setPan_(pan_to_point);
}

void PathologyViewer::modifyZoom(const qreal x, float num_steps, const QPointF& view_center, const QPointF& scene_center)
{
	qreal factor		= 1.0 + qreal(num_steps) * x / 300.;
	float maxDownsample = 1. / _sceneScale;
	QRectF FOV			= this->mapToScene(this->rect()).boundingRect();
	QRectF FOVImage		= QRectF(FOV.left() / _sceneScale, FOV.top() / _sceneScale, FOV.width() / _sceneScale, FOV.height() / _sceneScale);

	float scaleX	= static_cast<float>(m_instance_->document->image().getDimensions()[0]) / FOVImage.width();
	float scaleY	= static_cast<float>(m_instance_->document->image().getDimensions()[1]) / FOVImage.height();
	float minScale	= scaleX > scaleY ? scaleY : scaleX;
	float maxScale	= scaleX > scaleY ? scaleX : scaleY;
	if ((factor < 1.0 && maxScale < 0.5) || (factor > 1.0 && minScale > 2 * maxDownsample))
	{
		return;
	}

	this->modifyZoom(view_center, scene_center, factor);
}

void PathologyViewer::modifyZoom(const QPointF& zoom_view, const QPointF& zoom_scene, const qreal zoom_factor)
{
	if (zoom_view != m_instance_->view_state.zoom_view_center ||
		zoom_scene != m_instance_->view_state.zoom_scene_center ||
		zoom_factor != 1.0f)
	{
		setZoom_(zoom_view, zoom_scene, m_instance_->view_state.zoom_factor * zoom_factor);
	}
}

void PathologyViewer::setViewState(const PathologyViewState& state)
{
	this->setZoom_(state.zoom_view_center, state.zoom_scene_center, state.zoom_factor);
	this->setPan_(state.pan_position);
}

void PathologyViewer::refreshView(void)
{
	if (m_instance_)
	{
		QRectF fov = this->mapToScene(this->rect()).boundingRect();
		QRectF fov_image = QRectF(fov.left() / _sceneScale, fov.top() / _sceneScale, fov.width() / _sceneScale, fov.height() / _sceneScale);
		emit fieldOfViewChanged(fov_image, m_instance_->document->image().getBestLevelForDownSample((1.0f / _sceneScale) / this->transform().m11()));
		emit updateBBox(fov);
	}
}

void PathologyViewer::setPan_(const QPoint& pan_to_point)
{
	if (m_instance_)
	{
		QScrollBar *hBar	= horizontalScrollBar();
		QScrollBar *vBar	= verticalScrollBar();
		QPoint delta		= pan_to_point - m_instance_->view_state.pan_position;

		//_prevPan = pan_to_point;
		m_instance_->view_state.pan_position = pan_to_point;
		hBar->setValue(hBar->value() + (isRightToLeft() ? delta.x() : -delta.x()));
		vBar->setValue(vBar->value() - delta.y());
		this->refreshView();
	}
}

void PathologyViewer::setZoom_(const QPointF& zoom_view, const QPointF& zoom_scene, const qreal zoom_factor)
{
	if (m_instance_)
	{
		m_instance_->view_state.zoom_view_center	= zoom_view;
		m_instance_->view_state.zoom_scene_center	= zoom_scene;
		m_instance_->view_state.zoom_factor			= zoom_factor;

		this->resetMatrix();
		this->scale(zoom_factor, zoom_factor);
		this->centerOn(zoom_scene);

		QPointF delta_viewport_pos	= zoom_view - QPointF(width() / 2.0, height() / 2.0);
		QPointF viewport_center		= mapFromScene(zoom_scene) - delta_viewport_pos;
		centerOn(mapToScene(viewport_center.toPoint()));

		this->refreshView();
	}
}


void PathologyViewer::mouseMoveEvent(QMouseEvent* event)
{
	emit mouseMoveOccured(event);
}

void PathologyViewer::mousePressEvent(QMouseEvent* event)
{
	emit mousePressOccured(event);
}

void PathologyViewer::mouseReleaseEvent(QMouseEvent* event)
{
	emit mouseReleaseOccured(event);
}

void PathologyViewer::mouseDoubleClickEvent(QMouseEvent* event)
{
	emit mouseDoubleClickOccured(event);
}

void PathologyViewer::wheelEvent(QWheelEvent* event)
{
	emit wheelOccured(event);
}