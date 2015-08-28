#include "PathologyViewer.h"

#include <iostream>

#include <QResizeEvent>
#include <QApplication>
#include <QMenu>
#include <QMessageBox>
#include <QGLWidget>
#include <QTimeLine>
#include <QScrollBar>
#include <QHBoxLayout>

#include "MiniMap.h"
#include "RenderThread.h"
#include "PrefetchThread.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "interfaces/interfaces.h"
#include "core/PathologyEnums.h"
#include "WSITileGraphicsItem.h"
#include "WSITileGraphicsItemCache.h"

using std::vector;

PathologyViewer::PathologyViewer(QWidget *parent):
  QGraphicsView(parent),
  _img(NULL),
  _renderthread(NULL),
  _prefetchthread(NULL),
  _zoomSensitivity(0.5),
  _panSensitivity(0.5),
  _numScheduledScalings(0),
  _pan(false),
  _prevPan(0,0),
  _map(NULL), 
  _cache(NULL),
  _cacheSize(1000 * 1024 * 1024 * 3),
  _activeTool(NULL),
  _sceneScale(1.)
{
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setResizeAnchor(QGraphicsView::ViewportAnchor::AnchorViewCenter);
  setDragMode(QGraphicsView::DragMode::NoDrag);
  setContentsMargins(0,0,0,0);
  setAutoFillBackground(true);
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
}

PathologyViewer::~PathologyViewer()
{
  close();
  // Cleanup tools
  _activeTool = NULL;
  for (std::map<std::string, ToolPluginInterface*>::iterator it = _tools.begin(); it != _tools.end(); ++it) {
    delete it->second;
  }
  _tools.clear();
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
    emit fieldOfViewChanged(FOVImage, _img, _img->getBestLevelForDownSample(maxDownsample / this->transform().m11()), -1);
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
  if (_numScheduledScalings * numSteps < 0)  // if user moved the wheel in another direction, we reset previously scheduled scalings
    _numScheduledScalings = numSteps;

  QTimeLine *anim = new QTimeLine(300, this);
  anim->setUpdateInterval(5);

  connect(anim, SIGNAL(valueChanged(qreal)), SLOT(scalingTime(qreal)));
  connect(anim, SIGNAL(finished()), SLOT(zoomFinished()));
  anim->start();
}

void PathologyViewer::scalingTime(qreal x)
{
  qreal factor = 1.0 + qreal(_numScheduledScalings) / 300.0;
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
  emit fieldOfViewChanged(FOVImage, _img, _img->getBestLevelForDownSample((1. / this->_sceneScale) / this->transform().m11()), -1);
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
  emit fieldOfViewChanged(FOVImage, _img, _img->getBestLevelForDownSample(maxDownsample / this->transform().m11()), -1);
  emit updateBBox(FOV);
}

void PathologyViewer::addTool(ToolPluginInterface* tool) {
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
    _activeTool = _tools[toolName];
  }
}

void PathologyViewer::changeActiveTool() {
  if (sender()) {
    QAction* button = qobject_cast< QAction*>(sender());
    ToolPluginInterface* newActiveTool = _tools[button->objectName().toStdString()];
    if (newActiveTool) {
      _activeTool = newActiveTool;
    }
    else {
      _activeTool = NULL;
    }
  }
}

void PathologyViewer::onFieldOfViewChanged(const QRectF& FOV, MultiResolutionImage* img, const unsigned int level, int channel) {
  _renderthread->clearJobs();
}

void PathologyViewer::initialize(MultiResolutionImage *img) {
  close();
  setEnabled(true);
  _img = img;
  _renderthread = new RenderThread(img);
  unsigned int tileSize = 1024;

  unsigned int lastLevel = _img->getNumberOfLevels() - 1;
  for (int i = lastLevel; i >= 0; --i) {
    std::vector<unsigned long long> lastLevelDimensions = _img->getLevelDimensions(i);
    if (lastLevelDimensions[0] > tileSize && lastLevelDimensions[1] > tileSize) {
      lastLevel = i;
      break;
    }
  }
  initializeImage(scene(), tileSize, lastLevel);
  initializeMiniMap(lastLevel);
  setMouseTracking(true);
  QObject::connect(this, SIGNAL(channelChanged(int)), _renderthread, SLOT(onChannelChanged(int)));  
  QObject::connect(this, SIGNAL(fieldOfViewChanged(const QRectF, MultiResolutionImage*, const unsigned int, int)), this, SLOT(onFieldOfViewChanged(const QRectF, MultiResolutionImage*, const unsigned int, int)));
}

void PathologyViewer::onForegroundImageChanged(MultiResolutionImage* for_img) {
  if (_renderthread) {
    _renderthread->setForegroundImage(for_img);
    _cache->refresh();
  }
}

void PathologyViewer::setForegroundOpacity(const float& opacity) {
  if (_renderthread) {
    _renderthread->setForegroundOpacity(opacity);
    _cache->refresh();
  }
}

float PathologyViewer::getForegroundOpacity() const {
  if (_renderthread) {
    return _renderthread->getForegroundOpacity();
  }
  else {
    return 1;
  }
}

void PathologyViewer::initializeImage(QGraphicsScene* scn, unsigned int tileSize, unsigned int lastLevel) {  
  unsigned int nrLevels = _img->getNumberOfLevels();
  std::vector<unsigned long long> lastLevelDimensions = _img->getLevelDimensions(lastLevel);
  float lastLevelWidth = ((lastLevelDimensions[0] / tileSize) + 1) * tileSize;
  float lastLevelHeight = ((lastLevelDimensions[1] / tileSize) + 1) * tileSize;
  float longest = lastLevelWidth > lastLevelHeight ? lastLevelWidth : lastLevelHeight;
  _sceneScale = 1./_img->getLevelDownsample(lastLevel);
  for (float i = 0; i < lastLevelWidth; i += tileSize) {
    for (float j = 0; j < lastLevelHeight; j += tileSize) {
      WSITileGraphicsItem* item = new WSITileGraphicsItem(tileSize, lastLevel, lastLevel, _img, _renderthread, _cache);
      item->setPos(i + tileSize / 2., j + tileSize / 2.);
      item->setZValue(1. / ((float)lastLevel+1));
      scn->addItem(item);
      std::stringstream ss;
      ss << item->pos().x() << "_" << item->pos().y() << "_" << lastLevel;
      std::string key;
      ss >> key;
      _cache->set(key, item, tileSize*tileSize*_img->getSamplesPerPixel(), true);
      item->loadTile();
      item->loadNextLevel(true);
    }
  }
  while (_renderthread->numberOfJobs() > 0) {
  }

  QRectF n((lastLevelDimensions[0] / 2) - 1.5*longest, (lastLevelDimensions[1] / 2) - 1.5*longest, 3 * longest, 3 * longest);
  this->setSceneRect(n);
  this->fitInView(QRectF(0, 0, lastLevelDimensions[0], lastLevelDimensions[1]), Qt::AspectRatioMode::KeepAspectRatio);
}

void PathologyViewer::initializeMiniMap(unsigned int level) {
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
  QPixmap *ovPixMap = new QPixmap(QPixmap::fromImage(ovImg));
  delete[] overview;
  _map = new MiniMap(ovPixMap, this);
  if (this->layout()) {
    delete this->layout();
  }
  QHBoxLayout * Hlayout = new QHBoxLayout(this);
  QVBoxLayout * Vlayout = new QVBoxLayout();
  Hlayout->addStretch(4);
  Hlayout->setContentsMargins(30, 30, 30, 30);
  Hlayout->addLayout(Vlayout, 1);
  Vlayout->addStretch(4);
  Vlayout->addWidget(_map, 1);
  _map->updateFieldOfView(QRectF(0, 0, overviewDimensions[0], overviewDimensions[1]));
  QObject::connect(this, SIGNAL(updateBBox(const QRectF&)), _map, SLOT(updateFieldOfView(const QRectF&)));
  QObject::connect(_map, SIGNAL(positionClicked(QPointF)), this, SLOT(moveTo(const QPointF&)));
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
            emit channelChanged(i);
            _cache->refresh();
          }
        }
      }
    }
  }
}

void PathologyViewer::close() {
  if (_prefetchthread) {
    _prefetchthread->deleteLater();
    _prefetchthread = NULL;
  }
  scene()->clear();
  _cache->clear();
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
  emit fieldOfViewChanged(FOVImage, _img, _img->getBestLevelForDownSample(maxDownsample / this->transform().m11()), -1);
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