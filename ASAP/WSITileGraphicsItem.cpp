#include <iostream>
#include <QStyleOptionGraphicsItem>
#include <QPixmap>
#include <QPainter>
#include <QPointer>
#include <QGraphicsScene>
#include <QTransform>
#include "WSITileGraphicsItem.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "WSITileGraphicsItemCache.h"
#include "TileManager.h"
#include <QElapsedTimer>


WSITileGraphicsItem::WSITileGraphicsItem(QPixmap* item, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int itemLevel, unsigned int lastRenderLevel, const std::vector<float>& imgDownsamples, TileManager* manager, QPixmap* foregroundPixmap, ImageSource* foregroundTile, float foregroundOpacity, bool renderForeground) :
  QGraphicsItem(),
  _item(NULL),
  _manager(NULL),
  _tileX(tileX),
  _tileY(tileY),
  _tileSize(tileSize),
  _tileByteSize(tileByteSize),
  _itemLevel(itemLevel),
  _lastRenderLevel(lastRenderLevel),
  _foregroundPixmap(foregroundPixmap),
  _foregroundTile(foregroundTile),
  _foregroundOpacity(foregroundOpacity),
  _renderForeground(renderForeground)
{
  if (item) {
    _item = item;
  }
  if (manager) {
    _manager = manager;
  }
  _physicalSize = _tileSize / (imgDownsamples[_lastRenderLevel] / imgDownsamples[_itemLevel]);
  float lastRenderLevelDownsample = imgDownsamples[_lastRenderLevel];
  float itemLevelLOD = lastRenderLevelDownsample / imgDownsamples[_itemLevel];
  if (_itemLevel == _lastRenderLevel) {
    _lowerLOD = 0.;
  }
  else {
    float avgDownsample = (imgDownsamples[_itemLevel + 1] + imgDownsamples[_itemLevel]) / 2.;
    _lowerLOD = lastRenderLevelDownsample / avgDownsample;
  }
  if (_itemLevel == 0) {
    _upperLOD = std::numeric_limits<float>::max();
  }
  else {
    float avgDownsample = (imgDownsamples[_itemLevel - 1] + imgDownsamples[_itemLevel]) / 2.;
    _upperLOD = lastRenderLevelDownsample / avgDownsample;
  }
  this->setFlag(QGraphicsItem::ItemUsesExtendedStyleOption);
  _boundingRect = QRectF(-_physicalSize / 2., -_physicalSize / 2., _physicalSize, _physicalSize);
}

WSITileGraphicsItem::~WSITileGraphicsItem() {
  if (_item) {
    delete _item;
    _item = NULL;
  }
  if (_foregroundPixmap) {
    delete _foregroundPixmap;
    _foregroundPixmap = NULL;
  }
  if (_foregroundTile) {
    delete _foregroundTile;
    _foregroundTile = NULL;
  }
  if (_manager) {
    _manager = NULL;
  }
}

QRectF WSITileGraphicsItem::boundingRect() const{
  // return the size
  return _boundingRect;
}

void WSITileGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                                QWidget *widget){
  float lod = option->levelOfDetailFromTransform(painter->worldTransform());
  if (lod > _lowerLOD) {
    if (_item) {
      bool draw = false;
      if (lod <= _upperLOD) {
        draw = true;
      }
      else if (!_manager->isCovered(_itemLevel, _tileX, _tileY)) {
        draw = true;
      }
      if (draw) {
        QRectF pixmapArea = QRectF((option->exposedRect.left() + (_physicalSize / 2))*(_tileSize / _physicalSize), (option->exposedRect.top() + (_physicalSize / 2))*(_tileSize / _physicalSize), option->exposedRect.width()*(_tileSize / _physicalSize), option->exposedRect.height()*(_tileSize / _physicalSize));
        painter->drawPixmap(option->exposedRect, *_item, pixmapArea);
        if (_foregroundPixmap && _renderForeground && _foregroundOpacity > 0.0001) {
          painter->setOpacity(_foregroundOpacity);
          painter->drawPixmap(option->exposedRect, *_foregroundPixmap, pixmapArea);
        }
      }
    }
  }
}

void WSITileGraphicsItem::debugPrint() {
  std::cout << "Position (x,y): (" << this->pos().x() << ", "<< this->pos().y() << ")" << std::endl;
  std::cout << "Has pixmap: " << (_item ? "Yes" : "No") << std::endl;
  std::cout << "Visible: " << this->isVisible() << std::endl;
  std::cout << "Level: " << _itemLevel << std::endl;
  std::cout << "Bounding rectangle (x,y,w,h): (" << _boundingRect.x() << ", " << _boundingRect.y() << ", " << _boundingRect.width() << ", " << _boundingRect.height() << ")" << std::endl;
}

void WSITileGraphicsItem::setForegroundPixmap(QPixmap* foregroundPixmap) {
  QPixmap* oldPixmap = _foregroundPixmap;
  _foregroundPixmap = foregroundPixmap;
  delete oldPixmap;
  this->update();
}

ImageSource* WSITileGraphicsItem::getForegroundTile() {
  return _foregroundTile;
}

void WSITileGraphicsItem::setForegroundOpacity(float opacity) {
  _foregroundOpacity = opacity;
  this->update();
}

float WSITileGraphicsItem::getForegroundOpacity() {
  return _foregroundOpacity;
}

void WSITileGraphicsItem::setRenderForeground(bool renderForeground)
{
  _renderForeground = renderForeground;
  this->update();
}

bool WSITileGraphicsItem::getRenderForeground()
{
  return _renderForeground;
}
