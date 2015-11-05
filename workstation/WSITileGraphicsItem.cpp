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
#include "TileManager.h"
#include <QElapsedTimer>


WSITileGraphicsItem::WSITileGraphicsItem(QPixmap* item, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int itemLevel, unsigned int lastRenderLevel, const MultiResolutionImage* const img, TileManager* manager) :
  QGraphicsItem(),
  _item(NULL),
  _manager(NULL),
  _img(NULL),
  _tileX(tileX),
  _tileY(tileY),
  _tileSize(tileSize),
  _tileByteSize(tileByteSize),
  _itemLevel(itemLevel),
  _lastRenderLevel(lastRenderLevel)
{
  if (item) {
    _item = item;
  }
  if (manager) {
    _manager = manager;
  }
  if (img) {
    _img = img;
  }
  _physicalSize = _tileSize / (_img->getLevelDownsample(_lastRenderLevel) / _img->getLevelDownsample(_itemLevel));
  this->setFlag(QGraphicsItem::ItemUsesExtendedStyleOption);
  _boundingRect = QRectF(-_physicalSize / 2., -_physicalSize / 2., _physicalSize, _physicalSize);
}

WSITileGraphicsItem::~WSITileGraphicsItem() {
  if (_item) {
    delete _item;
    _item = NULL;
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
  float maxDownsample = _img->getLevelDownsample(_lastRenderLevel);
  unsigned int currentLevel = _img->getBestLevelForDownSample(maxDownsample / lod) > _lastRenderLevel ? _lastRenderLevel : _img->getBestLevelForDownSample(maxDownsample / lod);
  if (currentLevel <= _itemLevel) {
    if (_item) {
      if (!_manager->isCovered(_itemLevel, _tileX, _tileY) || currentLevel == _itemLevel) {
        QRectF pixmapArea = QRectF((option->exposedRect.left() + (_physicalSize / 2))*(_tileSize / _physicalSize), (option->exposedRect.top() + (_physicalSize / 2))*(_tileSize / _physicalSize), option->exposedRect.width()*(_tileSize / _physicalSize), option->exposedRect.height()*(_tileSize / _physicalSize));
        painter->drawPixmap(option->exposedRect, *_item, pixmapArea);
        /*
        if (_itemLevel == 0) {
        painter->setPen(QPen(QColor("white")));
        }
        else if (_itemLevel == 1) {
        painter->setPen(QPen(QColor("red")));
        }
        else if (_itemLevel == 2) {
        painter->setPen(QPen(QColor("blue")));
        }
        else if (_itemLevel == 3) {
        painter->setPen(QPen(QColor("green")));
        }
        else if (_itemLevel == 4) {
        painter->setPen(QPen(QColor("purple")));
        }
        else if (_itemLevel == 5) {
        painter->setPen(QPen(QColor("orange")));
        }
        else if (_itemLevel == 6) {
        painter->setPen(QPen(QColor("black")));
        }
        painter->drawRect(QRectF(this->boundingRect().left(), this->boundingRect().top(), _physicalSize, _physicalSize));
        QString location = QString("%1_%2_%3").arg(_tileX).arg(_tileY).arg(_itemLevel);
        QFont font = painter->font();
        font.setPointSizeF(2 * font.pointSizeF() * (_physicalSize / _tileSize));
        painter->setFont(font);
        painter->drawText(QPointF(0, 0), location);
        */
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