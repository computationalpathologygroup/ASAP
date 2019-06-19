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

PathologyViewer::PathologyViewer(QWidget *parent) :
	QGraphicsView(parent),
	_zoomSensitivity(0.5),
	_panSensitivity(0.5),
	_numScheduledScalings(0),
	_pan(false),
	_prevPan(0, 0),
	_cacheSize(1000 * 512 * 512 * 3),
	_sceneScale(1.0f),
	_renderthread(nullptr),
	_prefetchthread(nullptr),
	_map(nullptr),
	_cache(nullptr),
	_activeTool(nullptr),
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
  //setViewport(new QGLWidget());
  setViewportUpdateMode(ViewportUpdateMode::FullViewportUpdate);
  setInteractive(false);
  this->setScene(new QGraphicsScene); //Memleak!
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

unsigned long long PathologyViewer::getCacheSize(void) const {
  if (_cache) {
    return _cache->maxCacheSize();
  }
  else {
    return 0;
  }
}

void PathologyViewer::setCacheSize(unsigned long long& maxCacheSize) {
  if (_cache) {
    _cache->setMaxCacheSize(maxCacheSize);
  }
}

void PathologyViewer::resizeEvent(QResizeEvent *event) {
  QRect rect = QRect(QPoint(0, 0), event->size());
  QRectF FOV = this->mapToScene(rect).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / _sceneScale, FOV.top() / _sceneScale, FOV.width() / _sceneScale, FOV.height() / _sceneScale);
  float maxDownsample = 1. / _sceneScale;
  QGraphicsView::resizeEvent(event);

  if (m_instance_) {
    emit fieldOfViewChanged(FOVImage, m_instance_->document.image().getBestLevelForDownSample(maxDownsample / this->transform().m11()));
    emit updateBBox(FOV);
  }
}

void PathologyViewer::wheelEvent(QWheelEvent *event) {
  int numDegrees = event->delta() / 8;
  int numSteps = numDegrees / 15;  // see QWheelEvent documentation
  _zoomToScenePos = this->mapToScene(event->pos());
  _zoomToViewPos = event->pos();
  zoom(numSteps);
}

void PathologyViewer::zoom(float numSteps) {
  if (m_instance_)
  {
	  _numScheduledScalings += numSteps;
	  if (_numScheduledScalings * numSteps < 0) {
		  _numScheduledScalings = numSteps;
	  }

	  QTimeLine *anim = new QTimeLine(300, this);
	  anim->setUpdateInterval(5);

	  connect(anim, SIGNAL(valueChanged(qreal)), SLOT(scalingTime(qreal)));
	  connect(anim, SIGNAL(finished()), SLOT(zoomFinished()));
	  anim->start();
  }
}

void PathologyViewer::scalingTime(qreal x)
{
  qreal factor = 1.0 + qreal(_numScheduledScalings) * x / 300.;
  float maxDownsample = 1. / _sceneScale;
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / _sceneScale, FOV.top() / _sceneScale, FOV.width() / _sceneScale, FOV.height() / _sceneScale);
  float scaleX = static_cast<float>(m_instance_->document.image().getDimensions()[0]) / FOVImage.width();
  float scaleY = static_cast<float>(m_instance_->document.image().getDimensions()[1]) / FOVImage.height();
  float minScale = scaleX > scaleY ? scaleY : scaleX;
  float maxScale = scaleX > scaleY ? scaleX : scaleY;
  if ((factor < 1.0 && maxScale < 0.5) || (factor > 1.0 && minScale > 2*maxDownsample)) {
    return;
  }
  scale(factor, factor);
  centerOn(_zoomToScenePos);
  QPointF delta_viewport_pos = _zoomToViewPos - QPointF(width() / 2.0, height() / 2.0);
  QPointF viewport_center = mapFromScene(_zoomToScenePos) - delta_viewport_pos;
  centerOn(mapToScene(viewport_center.toPoint()));
  float tm11 = this->transform().m11();
  emit fieldOfViewChanged(FOVImage, m_instance_->document.image().getBestLevelForDownSample((1. / _sceneScale) / this->transform().m11()));
  emit updateBBox(FOV);
}

float PathologyViewer::getSceneScale(void) const
{
	return _sceneScale;
}

void PathologyViewer::zoomFinished()
{
  if (_numScheduledScalings > 0)
    _numScheduledScalings--;
  else
    _numScheduledScalings++;
  sender()->~QObject();
}

void PathologyViewer::moveTo(const QPointF& pos) {
  this->centerOn(pos);
  float maxDownsample = 1. / _sceneScale;
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / _sceneScale, FOV.top() / _sceneScale, FOV.width() / _sceneScale, FOV.height() / _sceneScale);
  emit fieldOfViewChanged(FOVImage, m_instance_->document.image().getBestLevelForDownSample(maxDownsample / this->transform().m11()));
  emit updateBBox(FOV);
}

void PathologyViewer::addTool(std::shared_ptr<ToolPluginInterface> tool) {
  if (tool) {
    _tools[tool->name()] = tool;
  }
}

bool PathologyViewer::hasTool(const std::string& toolName) const {
  if (_tools.find(toolName) != _tools.end()) {
    return true;
  }
  else {
    return false;
  }
}

void PathologyViewer::setActiveTool(const std::string& toolName) {
  if (_tools.find(toolName) != _tools.end()) {
    if (_activeTool) {
      _activeTool->setActive(false);
    }
    _activeTool = _tools[toolName];
    _activeTool->setActive(true);
  }
}

std::shared_ptr<ToolPluginInterface> PathologyViewer::getActiveTool() {
  return _activeTool;
}

void PathologyViewer::changeActiveTool() {
  if (sender()) {
    QAction* button = qobject_cast< QAction*>(sender());
    std::shared_ptr<ToolPluginInterface> newActiveTool = _tools[button->objectName().toStdString()];
    if (_activeTool && newActiveTool && _activeTool != newActiveTool) {
      _activeTool->setActive(false);
    }
    if (newActiveTool) {
      _activeTool = newActiveTool;
      _activeTool->setActive(true);
    }
    else {
      _activeTool = NULL;
    }
  }
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

  // 
  QRect current_fov			= m_instance_->current_fov;
  uint64_t current_level	= m_instance_->current_level;

  // Setups internals
  _cache = new WSITileGraphicsItemCache();
  _cache->setMaxCacheSize(_cacheSize);
  _renderthread = new RenderThread(this);
  _renderthread->setBackgroundImage(m_instance_->document.GetImage());
  _manager = new TileManager(*m_instance_, _renderthread, _cache, scene());
  setMouseTracking(true);
  std::vector<RenderWorker*> workers = _renderthread->getWorkers();
  for (int i = 0; i < workers.size(); ++i) {
    QObject::connect(workers[i], SIGNAL(tileLoaded(QPixmap*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)), _manager, SLOT(onTileLoaded(QPixmap*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)));
  }

  // Initializes GUI elements
  ASAP::TileInformation tile_info = m_instance_->document.GetTileInformation();

  initializeImage(scene(), tile_info.tile_size, tile_info.top_level);
  initializeGUIComponents(tile_info.top_level);
  QObject::connect(this, SIGNAL(backgroundChannelChanged(int)), _renderthread, SLOT(onBackgroundChannelChanged(int)));
  QObject::connect(_cache, SIGNAL(itemEvicted(WSITileGraphicsItem*)), _manager, SLOT(onTileRemoved(WSITileGraphicsItem*)));
  QObject::connect(this, SIGNAL(fieldOfViewChanged(const QRectF, const unsigned int)), this, SLOT(onFieldOfViewChanged(const QRectF, const unsigned int)));

  QRectF FOV = this->mapToScene(current_fov).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / _sceneScale, FOV.top() / _sceneScale, FOV.width() / _sceneScale, FOV.height() / _sceneScale);
  
  
  emit fieldOfViewChanged(QRectF(_manager->tileCoordinatesToPixelCoordinates(current_fov.topLeft(), current_level), _manager->tileCoordinatesToPixelCoordinates(current_fov.bottomRight(), current_level)), current_level);
  //_manager->loadTilesForFieldOfView(FOVImage, current_level);
}

void PathologyViewer::onForegroundImageChanged(std::weak_ptr<MultiResolutionImage> for_img, float scale) {
  _for_img = for_img;
  if (_renderthread) {
    _renderthread->setForegroundImage(_for_img, scale);
    _manager->refresh();
  }
}

void PathologyViewer::setForegroundLUT(const std::string& LUTname) {
  if (_renderthread) {
    _renderthread->onLUTChanged(LUTname);
    if (_for_img.lock()) {
      _manager->refresh();
    }
  }
}

void PathologyViewer::setForegroundWindowAndLevel(const float& window, const float& level) {
  if (_renderthread) {
    _renderthread->onWindowAndLevelChanged(window, level);
    if (_for_img.lock()) {
      _manager->refresh();
    }
  }
}

void PathologyViewer::setForegroundChannel(unsigned int channel) {
  if (_renderthread) {
    _renderthread->onForegroundChannelChanged(channel);
    if (_for_img.lock()) {
      _manager->refresh();
    }
  }
}


void PathologyViewer::setForegroundOpacity(const float& opacity) {
  if (_renderthread) {
    _renderthread->setForegroundOpacity(opacity);
    if (_for_img.lock()) {
      _manager->refresh();
    }
  }
}


float PathologyViewer::getForegroundOpacity() const {
  return _opacity;
}

void PathologyViewer::initializeImage(QGraphicsScene* scn, unsigned int tileSize, unsigned int lastLevel) {  
  unsigned int nrLevels = m_instance_->document.image().getNumberOfLevels();
  std::vector<unsigned long long> lastLevelDimensions = m_instance_->document.image().getLevelDimensions(lastLevel);
  float lastLevelWidth = ((lastLevelDimensions[0] / tileSize) + 1) * tileSize;
  float lastLevelHeight = ((lastLevelDimensions[1] / tileSize) + 1) * tileSize;
  float longest = lastLevelWidth > lastLevelHeight ? lastLevelWidth : lastLevelHeight;
  _sceneScale = 1. / m_instance_->document.image().getLevelDownsample(lastLevel);
  QRectF n((lastLevelDimensions[0] / 2) - 1.5*longest, (lastLevelDimensions[1] / 2) - 1.5*longest, 3 * longest, 3 * longest);
  this->setSceneRect(n);
  this->fitInView(QRectF(0, 0, lastLevelDimensions[0], lastLevelDimensions[1]), Qt::AspectRatioMode::KeepAspectRatio);

  _manager->loadAllTilesForLevel(lastLevel);
  float maxDownsample = 1. / _sceneScale;
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / _sceneScale, FOV.top() / _sceneScale, FOV.width() / _sceneScale, FOV.height() / _sceneScale);
  emit fieldOfViewChanged(FOVImage, m_instance_->document.image().getBestLevelForDownSample(maxDownsample / this->transform().m11()));
  while (_renderthread->numberOfJobs() > 0) {
  }
}

void PathologyViewer::initializeGUIComponents(unsigned int level) {
  // Initialize the minimap
  std::vector<unsigned long long> overviewDimensions = m_instance_->document.image().getLevelDimensions(level);
  unsigned int size = overviewDimensions[0] * overviewDimensions[1] * m_instance_->document.image().getSamplesPerPixel();
  unsigned char* overview = new unsigned char[size];
  m_instance_->document.image().getRawRegion<unsigned char>(0, 0, overviewDimensions[0], overviewDimensions[1], level, overview);
  QImage ovImg;
  if (m_instance_->document.image().getColorType() == pathology::ARGB) {
    ovImg = QImage(overview, overviewDimensions[0], overviewDimensions[1], overviewDimensions[0] * 4, QImage::Format_ARGB32).convertToFormat(QImage::Format_RGB888);
  }
  else if (m_instance_->document.image().getColorType() == pathology::RGB) {
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
  std::vector<double> spacing = m_instance_->document.image().getSpacing();
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
    if (m_instance_->document.image().getColorType() == pathology::ColorType::Indexed) {
      for (int i = 0; i < m_instance_->document.image().getSamplesPerPixel(); ++i) {
        rightClickMenu.addAction(QString("Channel ") + QString::number(i+1));
      }
      QAction* selectedItem = rightClickMenu.exec(globalPos);
      if (selectedItem)
      {
        for (int i = 0; i < m_instance_->document.image().getSamplesPerPixel(); ++i) {
          if (selectedItem->text() == QString("Channel ") + QString::number(i + 1)) {
            emit backgroundChannelChanged(i);
            _manager->refresh();
          }
        }
      }
    }
    else if (m_instance_->document.image().getNumberOfZPlanes() > 1) {
      for (int i = 0; i < m_instance_->document.image().getNumberOfZPlanes(); ++i) {
        rightClickMenu.addAction(QString("Plane ") + QString::number(i + 1));
      }
      QAction* selectedItem = rightClickMenu.exec(globalPos);
      if (selectedItem)
      {
        for (int i = 0; i < m_instance_->document.image().getNumberOfZPlanes(); ++i) {
          if (selectedItem->text() == QString("Plane ") + QString::number(i + 1)) {
			  m_instance_->document.image().setCurrentZPlaneIndex(i);
            _manager->refresh();
          }
        }
      }
    }
  }
}

void PathologyViewer::close() {
  if (this->window()) {
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
  if (_cache) {
    _cache->clear();
    delete _cache;
    _cache = NULL;
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
}

void PathologyViewer::togglePan(bool pan, const QPoint& startPos) {
  if (pan) {
    if (_pan) {
      return;
    }
    _pan = true;
    _prevPan = startPos;
    setCursor(Qt::ClosedHandCursor);
  }
  else {
    if (!_pan) {
      return;
    }
    _pan = false;
    _prevPan = QPoint(0, 0);
    setCursor(Qt::ArrowCursor);
  }
}

void PathologyViewer::pan(const QPoint& panTo) {
  QScrollBar *hBar = horizontalScrollBar();
  QScrollBar *vBar = verticalScrollBar();
  QPoint delta = panTo - _prevPan;
  _prevPan = panTo;
  hBar->setValue(hBar->value() + (isRightToLeft() ? delta.x() : -delta.x()));
  vBar->setValue(vBar->value() - delta.y());
  float maxDownsample = 1. / _sceneScale;
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / _sceneScale, FOV.top() / _sceneScale, FOV.width() / _sceneScale, FOV.height() / _sceneScale);
  emit fieldOfViewChanged(FOVImage, m_instance_->document.image().getBestLevelForDownSample(maxDownsample / this->transform().m11()));
  emit updateBBox(FOV);
}

void PathologyViewer::updateCurrentFieldOfView() {
  float maxDownsample = 1. / _sceneScale;
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / _sceneScale, FOV.top() / _sceneScale, FOV.width() / _sceneScale, FOV.height() / _sceneScale);
  emit fieldOfViewChanged(FOVImage, m_instance_->document.image().getBestLevelForDownSample(maxDownsample / this->transform().m11()));
  emit updateBBox(FOV);
}

void PathologyViewer::mousePressEvent(QMouseEvent *event)
{
  if (event->button() == Qt::MiddleButton)
  {
    togglePan(true, event->pos());
    event->accept();
    return;
  }
  if (_activeTool && event->button() == Qt::LeftButton) {
    _activeTool->mousePressEvent(event);
    if (event->isAccepted()) {
      return;
    }
  }
  event->ignore();
}

void PathologyViewer::mouseReleaseEvent(QMouseEvent *event)
{
  if (event->button() == Qt::MiddleButton)
  {
    togglePan(false);
    event->accept();
    return;
  }
  if (_activeTool && event->button() == Qt::LeftButton) {
    _activeTool->mouseReleaseEvent(event);
    if (event->isAccepted()) {
      return;
    }
  }
  event->ignore();
}

void PathologyViewer::mouseMoveEvent(QMouseEvent *event)
{
  QPointF imgLoc = this->mapToScene(event->pos()) / _sceneScale;
	// TODO: Clean this up or implement a more elegant selection method.
  qobject_cast<QMainWindow*>(this->parentWidget()->parentWidget()->parentWidget())->statusBar()->showMessage(QString("Current position in image coordinates: (") + QString::number(imgLoc.x()) + QString(", ") + QString::number(imgLoc.y()) + QString(")"), 1000);
  if (this->_pan) {
    pan(event->pos());
    event->accept();
    return;
  }
  if (_activeTool) {
    _activeTool->mouseMoveEvent(event);
    if (event->isAccepted()) {
      return;
    }
  }
  event->ignore();
}

void PathologyViewer::mouseDoubleClickEvent(QMouseEvent *event) {
  event->ignore();
  if (_activeTool) {
    _activeTool->mouseDoubleClickEvent(event);
  }
}

void PathologyViewer::keyPressEvent(QKeyEvent *event) {
  event->ignore();
  if (_activeTool) {
    _activeTool->keyPressEvent(event);
  }
}

bool PathologyViewer::isPanning() {
  return _pan;
}

void PathologyViewer::setPanSensitivity(float panSensitivity) {
      if (panSensitivity > 1) {
        _panSensitivity = 1;
      } else if (panSensitivity < 0.01) {
        _panSensitivity = 0.01;
      } else {
        _panSensitivity = panSensitivity;
      }
    };

float PathologyViewer::getPanSensitivity() const {
  return _panSensitivity;
};

void PathologyViewer::setZoomSensitivity(float zoomSensitivity) {
      if (zoomSensitivity > 1) {
        _zoomSensitivity = 1;
      } else if (zoomSensitivity < 0.01) {
        _zoomSensitivity = 0.01;
      } else {
        _zoomSensitivity = zoomSensitivity;
      }
    };

float PathologyViewer::getZoomSensitivity() const {
  return _zoomSensitivity;
};