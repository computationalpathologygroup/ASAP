#ifndef WSITileGraphicsItem_H
#define WSITileGraphicsItem_H

#include <QGraphicsItem>

class TileManager;
class MultiResolutionImage;

class WSITileGraphicsItem : public QGraphicsItem {
public:
  // make sure to set `item` to NULL in the constructor
  WSITileGraphicsItem(QPixmap* item, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int itemLevel, unsigned int lastRenderLevel, const MultiResolutionImage* const img, TileManager* manager);
  ~WSITileGraphicsItem();

  // you will need to add a destructor
  // (and probably a copy constructor and assignment operator)

  QRectF boundingRect() const;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget);

  void debugPrint();

private:
  // you'll probably want to store information about where you're
  // going to load the pixmap from, too

  QPixmap *_item;
  float _physicalSize;
  unsigned int _itemLevel;
  unsigned int _tileX;
  unsigned int _tileY;
  unsigned int _tileSize;
  unsigned int _tileByteSize;
  unsigned int _lastRenderLevel;
  QRectF _boundingRect;
  TileManager* _manager;
  const MultiResolutionImage* _img;

};

#endif