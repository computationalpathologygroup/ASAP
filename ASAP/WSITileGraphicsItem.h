#ifndef WSITileGraphicsItem_H
#define WSITileGraphicsItem_H

#include <QGraphicsItem>
#include <memory>

class TileManager;
class MultiResolutionImage;

class WSITileGraphicsItem : public QGraphicsItem {
public:
  // make sure to set `item` to NULL in the constructor
  WSITileGraphicsItem(QPixmap* item, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int itemLevel, unsigned int lastRenderLevel, const std::vector<float>& imgDownsamples, TileManager* manager);
  ~WSITileGraphicsItem();

  // you will need to add a destructor
  // (and probably a copy constructor and assignment operator)

  QRectF boundingRect() const;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget);

  void debugPrint();
  unsigned int getTileX() { return _tileX; }
  unsigned int getTileY() { return _tileY; }
  unsigned int getTileLevel() { return _itemLevel; }
  unsigned int getTileSize() { return _tileSize; }

private:
  // you'll probably want to store information about where you're
  // going to load the pixmap from, too

  QPixmap *_item;
  float _physicalSize;
  float _upperLOD;
  float _lowerLOD;
  unsigned int _itemLevel;
  unsigned int _tileX;
  unsigned int _tileY;
  unsigned int _tileSize;
  unsigned int _tileByteSize;
  unsigned int _lastRenderLevel;
  QRectF _boundingRect;
  TileManager* _manager;

};

#endif