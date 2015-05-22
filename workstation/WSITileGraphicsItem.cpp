#include <iostream>
#include <QStyleOptionGraphicsItem>
#include <QPixmap>
#include <QPainter>
#include <QPointer>
#include <QGraphicsScene>
#include <QTransform>
#include "WSITileGraphicsItem.h"
#include "RenderThread.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "WSITileGraphicsItemCache.h"
#include <QElapsedTimer>


WSITileGraphicsItem::WSITileGraphicsItem(unsigned int tileSize, float itemLevel, float lastRenderLevel, MultiResolutionImage* img, RenderThread* renderThread, WSITileGraphicsItemCache* cache) :
  QGraphicsItem(),
  QObject(),
  _item(NULL),
  _tileSize(tileSize),
  _itemLevel(itemLevel),
  _lastRenderLevel(lastRenderLevel),
  _img(img),
  _renderThread(renderThread),
  _cache(cache),
  _bottomLeft(NULL),
  _bottomRight(NULL),
  _topLeft(NULL),
  _topRight(NULL)
{
  if (_img) {
    this->setFlag(QGraphicsItem::ItemUsesExtendedStyleOption);
    _maxDownsample = _img->getLevelDownsample(_lastRenderLevel);
    _physicalSize = _tileSize / static_cast<float>(pow(2, _lastRenderLevel - _itemLevel));
    _boundingRect = QRectF(-_physicalSize / 2., -_physicalSize / 2., _physicalSize, _physicalSize);
    _downsample = _img->getLevelDownsample(_itemLevel);
  }
}

WSITileGraphicsItem::~WSITileGraphicsItem() {
  _renderThread->_senderDeletionMutex.lock();
  this->scene()->update();
  if (_item) {
    delete _item;
    _item = NULL;
  }
  _renderThread->_senderDeletionMutex.unlock();
}

QRectF WSITileGraphicsItem::boundingRect() const{
  // return the size
  return _boundingRect;
}

void WSITileGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                                QWidget *widget){
  float lod = option->levelOfDetailFromTransform(painter->worldTransform());
  unsigned int requestedLevel = _img->getBestLevelForDownSample(_maxDownsample / lod) > _lastRenderLevel ? _lastRenderLevel : _img->getBestLevelForDownSample(_maxDownsample / lod);
  if (requestedLevel <= _itemLevel) {
    if (NULL == _item){
      QRectF brect = _boundingRect;
      brect.translate(this->pos());
      long long imgPosX = brect.left() * _maxDownsample;
      long long imgPosY = brect.top() * _maxDownsample;
      this->_renderThread->addJob(_tileSize, _img->getSamplesPerPixel(), imgPosX, imgPosY, _itemLevel, QPointer<WSITileGraphicsItem>(this));
      this->setVisible(false);
    }
    if (requestedLevel < _itemLevel) {
      if (!_bottomLeft) {
        unsigned int size;
        _cache->get(generateKey() + "tl", _topLeft, size);
        if (!_topLeft) {
          _topLeft = new WSITileGraphicsItem(_tileSize, _itemLevel - 1, _lastRenderLevel, _img, _renderThread, _cache);
          _cache->set(generateKey() + "tl", _topLeft, _tileSize*_tileSize*_img->getSamplesPerPixel());
          this->scene()->addItem(_topLeft);
        }
        _cache->get(generateKey() + "tr", _topRight, size);
        if (!_topRight) {
          _topRight = new WSITileGraphicsItem(_tileSize, _itemLevel - 1, _lastRenderLevel, _img, _renderThread, _cache);
          this->scene()->addItem(_topRight);
          _cache->set(generateKey() + "tr", _topRight, _tileSize*_tileSize*_img->getSamplesPerPixel());
        }
        _cache->get(generateKey() + "bl", _bottomLeft, size);
        if (!_bottomLeft) {
          _bottomLeft = new WSITileGraphicsItem(_tileSize, _itemLevel - 1, _lastRenderLevel, _img, _renderThread, _cache);
          this->scene()->addItem(_bottomLeft);
          _cache->set(generateKey() + "bl", _bottomLeft, _tileSize*_tileSize*_img->getSamplesPerPixel());
        }
        _cache->get(generateKey() + "br", _bottomRight, size);
        if (!_bottomRight) {
          _bottomRight = new WSITileGraphicsItem(_tileSize, _itemLevel - 1, _lastRenderLevel, _img, _renderThread, _cache);
          this->scene()->addItem(_bottomRight);
          _cache->set(generateKey() + "br", _bottomRight, _tileSize*_tileSize*_img->getSamplesPerPixel());
        }
        _topLeft->setPos(scenePos().x() - 0.25 * _physicalSize, scenePos().y() - 0.25 * _physicalSize);
        _topRight->setPos(scenePos().x() + 0.25 * _physicalSize, scenePos().y() - 0.25 * _physicalSize);
        _bottomLeft->setPos(scenePos().x() - 0.25 * _physicalSize, scenePos().y() + 0.25 * _physicalSize);
        _bottomRight->setPos(scenePos().x() + 0.25 * _physicalSize, scenePos().y() + 0.25 * _physicalSize);
        _topLeft->setZValue(1. / ((float)_itemLevel));
        _topRight->setZValue(1. / ((float)_itemLevel));
        _bottomLeft->setZValue(1. / ((float)_itemLevel));
        _bottomRight->setZValue(1. / ((float)_itemLevel));
      }
    }
    if (_item) {
      //painter->setClipRect(option->exposedRect.marginsAdded(QMargins(1, 1, 1, 1)));
      QRectF pixmapArea = QRectF((option->exposedRect.left() + (_physicalSize / 2))*(_tileSize / _physicalSize), (option->exposedRect.top() + (_physicalSize / 2))*(_tileSize / _physicalSize), option->exposedRect.width()*(_tileSize / _physicalSize), option->exposedRect.height()*(_tileSize / _physicalSize));
      painter->drawPixmap(option->exposedRect, *_item, pixmapArea);
      //painter->setPen(QPen(QColor("white")));
      //painter->drawRect(QRectF(this->boundingRect().left(), this->boundingRect().top(), _physicalSize, _physicalSize));
    }
  }
}

std::string WSITileGraphicsItem::generateKey() {
  std::stringstream ss;
  ss << this->pos().x() << "_" << this->pos().y() << "_" << _itemLevel;
  std::string key;
  ss >> key;
  return key;
}

void WSITileGraphicsItem::initializeItem(QPixmap* pmap) {
  if (_item) {
    delete _item;
    _item = NULL;
  }
  _item = pmap;
  this->setVisible(true);
}

void WSITileGraphicsItem::refreshItem() {
  if (_item) {
    delete _item;
    _item = NULL;
  }
  this->update();
}

void WSITileGraphicsItem::debugPrint() {
  std::cout << "Position (x,y): (" << this->pos().x() << ", "<< this->pos().y() << ")" << std::endl;
  std::cout << "Has pixmap: " << (_item ? "Yes" : "No") << std::endl;
  std::cout << "Visible: " << this->isVisible() << std::endl;
  std::cout << "Level: " << _itemLevel << std::endl;
  std::cout << "Bounding rectangle (x,y,w,h): (" << _boundingRect.x() << ", " << _boundingRect.y() << ", " << _boundingRect.width() << ", " << _boundingRect.height() << ")" << std::endl;
}