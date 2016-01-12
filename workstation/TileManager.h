#ifndef TileManager_H
#define TileManager_H

#include <QObject>
#include <QRectF>
#include <QPointF>
#include <QPointer>
#include <map>
#include <memory>

class MultiResolutionImage;
class RenderThread;
class WSITileGraphicsItemCache;
class WSITileGraphicsItem;
class QGraphicsScene;
class QPainterPath;

class TileManager : public QObject {
  Q_OBJECT

private:
  std::vector<float> _levelDownsamples;
  std::vector<std::vector<unsigned long long> > _levelDimensions;
  unsigned int _tileSize;
  QRect _lastFOV;
  unsigned int _lastLevel;
  unsigned int _lastRenderLevel;
  std::map<unsigned int, std::map<int, std::map<int, unsigned char> > > _coverage;
  QPointer<RenderThread> _renderThread;
  QPointer<WSITileGraphicsItemCache> _cache;
  QPointer<QGraphicsScene> _scene;
  std::vector<QPainterPath> _coverageMaps;
  bool _coverageMapCacheMode;
  
  QPoint pixelCoordinatesToTileCoordinates(QPointF coordinate, unsigned int level);
  QPointF tileCoordinatesToPixelCoordinates(QPoint coordinate, unsigned int level);
  QPoint getLevelTiles(unsigned int level);

  TileManager(const TileManager& that);

signals:
  void coverageUpdated();

public:
  // make sure to set `item` to NULL in the constructor
  TileManager(std::shared_ptr<MultiResolutionImage> img, unsigned int tileSize, unsigned int lastRenderLevel, RenderThread* renderThread, WSITileGraphicsItemCache* _cache, QGraphicsScene* scene);
  ~TileManager();

  void loadAllTilesForLevel(unsigned int level);
  void loadTilesForFieldOfView(const QRectF& FOV, const unsigned int level);

  void resetCoverage(unsigned int level);
  unsigned char providesCoverage(unsigned int level, int tile_x = -1, int tile_y = -1);
  bool isCovered(unsigned int level, int tile_x = -1, int tile_y = -1);
  void setCoverage(unsigned int level, int tile_x, int tile_y, unsigned char covers);
  std::vector<QPainterPath> getCoverageMaps();
  void setCoverageMapModeToCache();
  void setCoverageMapModeToVisited();
  
  void clear();
  void refresh();

public slots:
  void onTileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel);
  void onTileRemoved(WSITileGraphicsItem* tile);

};

#endif