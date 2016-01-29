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

#include "MiniMap.h"
#include "ScaleBar.h"
#include "RenderThread.h"
#include "PrefetchThread.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "interfaces/interfaces.h"
#include "core/PathologyEnums.h"
#include "WSITileGraphicsItem.h"
#include "WSITileGraphicsItemCache.h"
#include "TileManager.h"
#include "RenderWorker.h"

using std::vector;

PathologyViewer::PathologyViewer(QWidget *parent):
  QGraphicsView(parent),
  _renderthread(NULL),
  _prefetchthread(NULL),
  _zoomSensitivity(0.5),
  _panSensitivity(0.5),
  _numScheduledScalings(0),
  _pan(false),
  _prevPan(0, 0),
  _map(NULL),
  _cache(NULL),
  _cacheSize(1000 * 512 * 512 * 3),
  _activeTool(NULL),
  _sceneScale(1.),
  _manager(NULL),
  _scaleBar(NULL)
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
  this->setScene(new QGraphicsScene);
  this->setBackgroundBrush(QBrush(QColor(252, 252, 252)));
  this->scene()->setBackgroundBrush(QBrush(QColor(252, 252, 252)));
  this->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(this, SIGNAL(customContextMenuRequested(const QPoint&)),
          this, SLOT(showContextMenu(const QPoint&)));
  _cache = new WSITileGraphicsItemCache();
  _cache->setMaxCacheSize(_cacheSize);
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

unsigned long long PathologyViewer::getCacheSize() {
  if (_cache) {
    return _cache->maxCacheSize();
  }
  else {
    return 0;
  }
}

void PathologyViewer::setCacheSize(unsigned long long& maxCacheSize) {
  if (_cache) {
    return _cache->setMaxCacheSize(maxCacheSize);
  }
}

void PathologyViewer::resizeEvent(QResizeEvent *event) {
  QRect rect = QRect(QPoint(0, 0), event->size());
  QRectF FOV = this->mapToScene(rect).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / this->_sceneScale, FOV.top() / this->_sceneScale, FOV.width() / this->_sceneScale, FOV.height() / this->_sceneScale);
  float maxDownsample = 1. / this->_sceneScale;
  QGraphicsView::resizeEvent(event);
  if (_img) {    
    emit fieldOfViewChanged(FOVImage, _img->getBestLevelForDownSample(maxDownsample / this->transform().m11()));
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
  if (!_img) {
    return;
  }
  _numScheduledScalings += numSteps;
  if (_numScheduledScalings * numSteps < 0)  {
    _numScheduledScalings = numSteps;
  }

  QTimeLine *anim = new QTimeLine(300, this);
  anim->setUpdateInterval(5);

  connect(anim, SIGNAL(valueChanged(qreal)), SLOT(scalingTime(qreal)));
  connect(anim, SIGNAL(finished()), SLOT(zoomFinished()));
  anim->start();
}

void PathologyViewer::scalingTime(qreal x)
{
  qreal factor = 1.0 + qreal(_numScheduledScalings) * x / 300.;
  float maxDownsample = 1. / this->_sceneScale;
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / this->_sceneScale, FOV.top() / this->_sceneScale, FOV.width() / this->_sceneScale, FOV.height() / this->_sceneScale);
  float scaleX = static_cast<float>(_img->getDimensions()[0]) / FOVImage.width();
  float scaleY = static_cast<float>(_img->getDimensions()[1]) / FOVImage.height();
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
  emit fieldOfViewChanged(FOVImage, _img->getBestLevelForDownSample((1. / this->_sceneScale) / this->transform().m11()));
  emit updateBBox(FOV);
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
  float maxDownsample = 1. / this->_sceneScale;
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / this->_sceneScale, FOV.top() / this->_sceneScale, FOV.width() / this->_sceneScale, FOV.height() / this->_sceneScale);
  emit fieldOfViewChanged(FOVImage, _img->getBestLevelForDownSample(maxDownsample / this->transform().m11()));
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

void PathologyViewer::initialize(std::shared_ptr<MultiResolutionImage> img) {
  close();
  setEnabled(true);
  _img = img;
  unsigned int tileSize = 512;
  unsigned int lastLevel = _img->getNumberOfLevels() - 1;
  for (int i = lastLevel; i >= 0; --i) {
    std::vector<unsigned long long> lastLevelDimensions = _img->getLevelDimensions(i);
    if (lastLevelDimensions[0] > tileSize && lastLevelDimensions[1] > tileSize) {
      lastLevel = i;
      break;
    }
  }
  _renderthread = new RenderThread(this);
  _renderthread->setBackgroundImage(img);
  _manager = new TileManager(_img, tileSize, lastLevel, _renderthread, _cache, scene());
  setMouseTracking(true);
  std::vector<RenderWorker*> workers = _renderthread->getWorkers();
  for (int i = 0; i < workers.size(); ++i) {
    QObject::connect(workers[i], SIGNAL(tileLoaded(QPixmap*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)), _manager, SLOT(onTileLoaded(QPixmap*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)));
  }
  initializeImage(scene(), tileSize, lastLevel);
  initializeGUIComponents(lastLevel);
  QObject::connect(this, SIGNAL(backgroundChannelChanged(int)), _renderthread, SLOT(onBackgroundChannelChanged(int)));
  QObject::connect(_cache, SIGNAL(itemEvicted(WSITileGraphicsItem*)), _manager, SLOT(onTileRemoved(WSITileGraphicsItem*)));
  QObject::connect(this, SIGNAL(fieldOfViewChanged(const QRectF, const unsigned int)), this, SLOT(onFieldOfViewChanged(const QRectF, const unsigned int)));
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / this->_sceneScale, FOV.top() / this->_sceneScale, FOV.width() / this->_sceneScale, FOV.height() / this->_sceneScale);
  emit fieldOfViewChanged(FOVImage, _img->getBestLevelForDownSample((1. / this->_sceneScale) / this->transform().m11()));
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
  unsigned int nrLevels = _img->getNumberOfLevels();
  std::vector<unsigned long long> lastLevelDimensions = _img->getLevelDimensions(lastLevel);
  float lastLevelWidth = ((lastLevelDimensions[0] / tileSize) + 1) * tileSize;
  float lastLevelHeight = ((lastLevelDimensions[1] / tileSize) + 1) * tileSize;
  float longest = lastLevelWidth > lastLevelHeight ? lastLevelWidth : lastLevelHeight;
  _sceneScale = 1. / _img->getLevelDownsample(lastLevel);
  QRectF n((lastLevelDimensions[0] / 2) - 1.5*longest, (lastLevelDimensions[1] / 2) - 1.5*longest, 3 * longest, 3 * longest);
  this->setSceneRect(n);
  this->fitInView(QRectF(0, 0, lastLevelDimensions[0], lastLevelDimensions[1]), Qt::AspectRatioMode::KeepAspectRatio);

  _manager->loadAllTilesForLevel(lastLevel);
  float maxDownsample = 1. / this->_sceneScale;
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / this->_sceneScale, FOV.top() / this->_sceneScale, FOV.width() / this->_sceneScale, FOV.height() / this->_sceneScale);
  emit fieldOfViewChanged(FOVImage, _img->getBestLevelForDownSample(maxDownsample / this->transform().m11()));
  while (_renderthread->numberOfJobs() > 0) {
  }
}

void PathologyViewer::initializeGUIComponents(unsigned int level) {
  // Initialize the minimap
  std::vector<unsigned long long> overviewDimensions = _img->getLevelDimensions(level);
  unsigned int size = overviewDimensions[0] * overviewDimensions[1] * _img->getSamplesPerPixel();
  unsigned char* overview = new unsigned char[size];
  _img->getRawRegion<unsigned char>(0, 0, overviewDimensions[0], overviewDimensions[1], level, overview);
  QImage ovImg;
  if (_img->getColorType() == pathology::ARGB) {
    ovImg = QImage(overview, overviewDimensions[0], overviewDimensions[1], overviewDimensions[0] * 4, QImage::Format_ARGB32).convertToFormat(QImage::Format_RGB888);
  }
  else if (_img->getColorType() == pathology::RGB) {
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
  std::vector<double> spacing = _img->getSpacing();
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

  if (_img) {
    QMenu rightClickMenu;
    if (_img->getColorType() == pathology::ColorType::Indexed) {
      for (int i = 0; i < _img->getSamplesPerPixel(); ++i) {
        rightClickMenu.addAction(QString("Channel ") + QString::number(i+1));
      }
      QAction* selectedItem = rightClickMenu.exec(globalPos);
      if (selectedItem)
      {
        for (int i = 0; i < _img->getSamplesPerPixel(); ++i) {
          if (selectedItem->text() == QString("Channel ") + QString::number(i + 1)) {
            emit backgroundChannelChanged(i);
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
  _cache->clear();
  if (_manager) {
    _manager->clear();
    delete _manager;
    _manager = NULL;
  }
  _img = NULL;
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
  float maxDownsample = 1. / this->_sceneScale;
  QRectF FOV = this->mapToScene(this->rect()).boundingRect();
  QRectF FOVImage = QRectF(FOV.left() / this->_sceneScale, FOV.top() / this->_sceneScale, FOV.width() / this->_sceneScale, FOV.height() / this->_sceneScale);
  emit fieldOfViewChanged(FOVImage, _img->getBestLevelForDownSample(maxDownsample / this->transform().m11()));
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