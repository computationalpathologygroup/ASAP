#ifndef TileManager_H
#define TileManager_H

#include <QObject>
#include <QRectF>
#include <QPointF>
#include <QPointer>
#include <map>
#include <memory>

class MultiResolutionImage;
class IOThread;
class WSITileGraphicsItemCache;
class WSITileGraphicsItem;
class QGraphicsScene;
class QPainterPath;
class ImageSource;
class QPixmap;

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
  QPointer<IOThread> _ioThread;
  QPointer<WSITileGraphicsItemCache> _cache;
  QPointer<QGraphicsScene> _scene;
  std::vector<QPainterPath> _coverageMaps;
  bool _coverageMapCacheMode;
  float _foregroundOpacity;
  bool _renderForeground;
  
  QPoint pixelCoordinatesToTileCoordinates(QPointF coordinate, unsigned int level);
  QPointF tileCoordinatesToPixelCoordinates(QPoint coordinate, unsigned int level);
  QPoint getLevelTiles(unsigned int level);

  TileManager(const TileManager& that);

signals:
  void coverageUpdated();

public:
  // make sure to set `item` to NULL in the constructor
  TileManager(std::shared_ptr<MultiResolutionImage> img, unsigned int tileSize, unsigned int lastRenderLevel, IOThread* renderThread, WSITileGraphicsItemCache* _cache, QGraphicsScene* scene);
  ~TileManager();

  void loadAllTilesForLevel(unsigned int level);
  void loadTilesForFieldOfView(const QRectF& FOV, const unsigned int level);

  void updateTileForegounds();

  void resetCoverage(unsigned int level);
  unsigned char providesCoverage(unsigned int level, int tile_x = -1, int tile_y = -1);
  bool isCovered(unsigned int level, int tile_x = -1, int tile_y = -1);
  void setCoverage(unsigned int level, int tile_x, int tile_y, unsigned char covers);
  std::vector<QPainterPath> getCoverageMaps();
  void setCoverageMapModeToCache();
  void setCoverageMapModeToVisited();
  
  void clear();
  void refresh();

  void reloadLastFOV();

public slots:
  void onForegroundTileRendered(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileLevel);
  void onTileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel, ImageSource* foregroundTile, QPixmap* foregroundPixmap);
  void onTileRemoved(WSITileGraphicsItem* tile);
  void onForegroundOpacityChanged(float opacity);
  void onRenderForegroundChanged(bool renderForeground);

};

#endif