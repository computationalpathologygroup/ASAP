#ifndef WSITileGraphicsItem_H
#define WSITileGraphicsItem_H

#include <QGraphicsItem>

class MultiResolutionImage;
class RenderThread;
class WSITileGraphicsItemCache;

class WSITileGraphicsItem : public QGraphicsItem, public QObject {
public:
  // make sure to set `item` to NULL in the constructor
  WSITileGraphicsItem(unsigned int tileSize, float itemLevel, float lastRenderLevel, MultiResolutionImage* img, RenderThread* renderThread, WSITileGraphicsItemCache* cache = NULL);
  ~WSITileGraphicsItem();

  // you will need to add a destructor
  // (and probably a copy constructor and assignment operator)

  QRectF boundingRect() const;
  void initializeItem(QPixmap* pmap);
  void refreshItem();

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget);

  void debugPrint();

private:
  // you'll probably want to store information about where you're
  // going to load the pixmap from, too

  QPixmap *_item;
  unsigned int _tileSize;
  float _itemLevel;
  float _lastRenderLevel;
  float _maxDownsample;
  float _physicalSize;
  float _downsample;
  QRectF _boundingRect;
  MultiResolutionImage *_img;
  RenderThread *_renderThread;
  WSITileGraphicsItemCache* _cache;

  QPointer<WSITileGraphicsItem> _topLeft;
  QPointer<WSITileGraphicsItem> _topRight;
  QPointer<WSITileGraphicsItem> _bottomLeft;
  QPointer<WSITileGraphicsItem> _bottomRight;

  std::string generateKey();
};

#endif