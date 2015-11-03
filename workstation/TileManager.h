#ifndef TileManager_H
#define TileManager_H

#include <QObject>
#include <QRectF>
#include <QPointF>
#include <map>

class MultiResolutionImage;
class RenderThread;
class WSITileGraphicsItemCache;
class QGraphicsScene;

class TileManager : public QObject {
  Q_OBJECT

private:
  MultiResolutionImage* _img;
  unsigned int _tileSize;
  QRect _lastFOV;
  unsigned int _lastLevel;
  std::map<unsigned int, std::map<int, std::map<int, unsigned char> > > _coverage;
  RenderThread* _renderThread;
  WSITileGraphicsItemCache* _cache;
  QGraphicsScene* _scene;
  
  QPoint pixelCoordinatesToTileCoordinates(QPointF coordinate, unsigned int level);


  TileManager(const TileManager& that);

public:
  // make sure to set `item` to NULL in the constructor
  TileManager(MultiResolutionImage* img, unsigned int tileSize, RenderThread* renderThread, WSITileGraphicsItemCache* _cache, QGraphicsScene* scene);
  ~TileManager();

  void loadTilesForFieldOfView(const QRectF& FOV, const unsigned int level, const int channel = -1);

  void resetCoverage(unsigned int level);
  unsigned char providesCoverage(unsigned int level, int tile_x = -1, int tile_y = -1);
  bool isCovered(unsigned int level, int tile_x = -1, int tile_y = -1);
  void setCoverage(unsigned int level, int tile_x, int tile_y, unsigned char covers);

public slots:
  void onTileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel, unsigned int lastRenderLevel);
  void onTileRemoved();

};

#endif