#ifndef TileManager_H
#define TileManager_H

#include <QObject>
#include <QRectF>
#include <QPointF>
#include <QPointer>
#include <map>
#include <memory>

#include "documents/DocumentInstance.h"

class MultiResolutionImage;
class RenderThread;
class WSITileGraphicsItemCache;
class WSITileGraphicsItem;
class QGraphicsScene;
class QPainterPath;

typedef std::map<uint32_t, std::map<int32_t, std::map<int32_t, uchar>>> CoverageMap;

class TileManager : public QObject {
  Q_OBJECT
	signals:
	  void coverageUpdated();

	public:
	  // make sure to set `item` to NULL in the constructor
	  TileManager(ASAP::DocumentInstance& document_instance, RenderThread* renderThread, WSITileGraphicsItemCache* _cache, QGraphicsScene* scene);
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
  
	  QPoint pixelCoordinatesToTileCoordinates(QPointF coordinate, unsigned int level);
	  QPointF tileCoordinatesToPixelCoordinates(QPoint coordinate, unsigned int level);

	  void clear();
	  void refresh();

	public slots:
	  void onTileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel);
	  void onTileRemoved(WSITileGraphicsItem* tile);

	private:
		CoverageMap					_coverage;
		ASAP::DocumentInstance&		m_instance_;
		const ASAP::TileInformation	m_tile_information_;
		QRect						m_last_loaded_FOV_;


		QPointer<RenderThread> _renderThread;
		QPointer<WSITileGraphicsItemCache> _cache;
		QPointer<QGraphicsScene> _scene;
		bool _coverageMapCacheMode;

		QPoint getLevelTiles(unsigned int level);

		TileManager(const TileManager& that);
		void updateCoverageMap_(const unsigned int level, const int tile_x, const int tile_y, const unsigned char covers);
};
#endif