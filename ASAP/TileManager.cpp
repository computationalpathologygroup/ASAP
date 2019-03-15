#include <sstream>
#include "TileManager.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "RenderThread.h"
#include "WSITileGraphicsItem.h"
#include "WSITileGraphicsItemCache.h"
#include <QGraphicsScene>
#include <QPainterPath>
#include <QCoreApplication>
#include <cmath>

TileManager::TileManager(ASAP::Document& document, unsigned int tileSize, unsigned int lastRenderLevel, RenderThread* renderThread, WSITileGraphicsItemCache* cache, QGraphicsScene* scene) :
m_tiled_document_(document),
m_tile_information_(document.AccessTileInformation()),
_renderThread(renderThread),
_cache(cache),
_scene(scene),
_coverageMaps(),
_coverageMapCacheMode(false)
{
	m_tile_information_.tile_size = tileSize;
	m_tile_information_.last_render_level = lastRenderLevel;
	for (unsigned int i = 0; i < m_tiled_document_.AccessImage().getNumberOfLevels(); ++i)
	{
		m_tile_information_.downsamples.push_back(m_tiled_document_.AccessImage().getLevelDownsample(i));
		m_tile_information_.dimensions.push_back(m_tiled_document_.AccessImage().getLevelDimensions(i));
	}
}

TileManager::~TileManager() {
  _renderThread = nullptr;
  _cache = nullptr;
  _scene = nullptr;
}

void TileManager::resetCoverage(unsigned int level)
{
	m_tile_information_.coverage[level] = std::map<int32_t, std::map<int32_t, uchar>>();
	if (_coverageMaps.size() > level)
	{
		_coverageMaps[level] = QPainterPath();
	}
}

QPoint TileManager::pixelCoordinatesToTileCoordinates(QPointF coordinate, unsigned int level) {
  if (level < m_tile_information_.downsamples.size()) {
    return QPoint(std::floor((coordinate.x() / m_tile_information_.downsamples[level]) / m_tile_information_.tile_size), std::floor((coordinate.y() / m_tile_information_.downsamples[level]) / m_tile_information_.tile_size));
  }
  else {
    return QPoint();
  }
}

QPointF TileManager::tileCoordinatesToPixelCoordinates(QPoint coordinate, unsigned int level) {  
  if (level < m_tile_information_.downsamples.size()) {
    return QPointF(coordinate.x() * m_tile_information_.downsamples[level] * m_tile_information_.tile_size, coordinate.y() * m_tile_information_.downsamples[level] * m_tile_information_.tile_size);
  }
  else {
    return QPoint();
  }
}

QPoint TileManager::getLevelTiles(unsigned int level) {
  if (level < m_tile_information_.dimensions.size()) {
    std::vector<unsigned long long> dims = m_tile_information_.dimensions[level];
    return QPoint(std::ceil(dims[0] / static_cast<float>(m_tile_information_.tile_size)), std::ceil(dims[1] / static_cast<float>(m_tile_information_.tile_size)));
  }
  else {
    return QPoint();
  }
}

void TileManager::loadAllTilesForLevel(unsigned int level) {
  if (_renderThread) {
    if (level < m_tile_information_.downsamples.size()) {
      std::vector<unsigned long long> baseLevelDims = m_tile_information_.dimensions[0];
      this->loadTilesForFieldOfView(QRectF(0, 0, baseLevelDims[0], baseLevelDims[1]), level);
    }
  }
}

void TileManager::loadTilesForFieldOfView(const QRectF& FOV, const unsigned int level) {
  if (level > m_tile_information_.last_render_level) {
    return;
  }
  if (_renderThread) {
    QPoint topLeftTile = this->pixelCoordinatesToTileCoordinates(FOV.topLeft(), level);
    QPoint bottomRightTile = this->pixelCoordinatesToTileCoordinates(FOV.bottomRight(), level);
    QRect FOVTile = QRect(topLeftTile, bottomRightTile);
    QPoint nrTiles = getLevelTiles(level);
    float levelDownsample = m_tile_information_.downsamples[level];
    if (FOVTile != m_tile_information_.last_FOV || level != m_tile_information_.last_level) {
      m_tile_information_.last_level = level;
      m_tile_information_.last_FOV = FOVTile;
      for (int x = topLeftTile.x(); x <= bottomRightTile.x(); ++x) {
        if (x >= 0 && x <= nrTiles.x()) {
          for (int y = topLeftTile.y(); y <= bottomRightTile.y(); ++y) {
            if (y >= 0 && y <= nrTiles.y() && providesCoverage(level, x, y) < 1)
			{
				setCoverage(level, x, y, 1);
				_renderThread->addJob(m_tile_information_.tile_size, x, y, level);
            }
          }
        }
      }
    }
  }
}

void TileManager::onTileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel) {
  WSITileGraphicsItem* item = new WSITileGraphicsItem(tile, tileX, tileY, tileSize, tileByteSize, tileLevel, m_tile_information_.last_render_level, m_tile_information_.downsamples, this);
  std::stringstream ss;
  ss << tileX << "_" << tileY << "_" << tileLevel;
  std::string key;
  ss >> key;
  if (_scene) {
    setCoverage(tileLevel, tileX, tileY, 2);
    float tileDownsample = m_tile_information_.downsamples[tileLevel];
    float maxDownsample = m_tile_information_.downsamples[m_tile_information_.last_render_level];
    float posX = (tileX * tileDownsample * tileSize) / maxDownsample + ((tileSize * tileDownsample) / (2 * maxDownsample));
    float posY = (tileY * tileDownsample * tileSize) / maxDownsample + ((tileSize * tileDownsample) / (2 * maxDownsample));
    _scene->addItem(item);
    item->setPos(posX, posY);
    item->setZValue(1. / ((float)tileLevel + 1.));
  }
  if (_cache) {
    _cache->set(key, item, tileByteSize, tileLevel == m_tile_information_.last_render_level);
  }
}

void TileManager::onTileRemoved(WSITileGraphicsItem* tile) {
  _scene->removeItem(tile);
  setCoverage(tile->getTileLevel(), tile->getTileX(), tile->getTileY(), 0);
  delete tile;
}

void TileManager::setCoverageMapModeToCache() {
  _coverageMapCacheMode = true;
}
void TileManager::setCoverageMapModeToVisited() {
  _coverageMapCacheMode = false;
}

unsigned char TileManager::providesCoverage(unsigned int level, int tile_x, int tile_y) {
  std::map<int32_t, std::map<int32_t, uchar>>& cover_level = m_tile_information_.coverage[level];
  if (cover_level.empty()) {
    return 0;
  }

  if (tile_x < 0 || tile_y < 0) {
    for (auto it_x = cover_level.begin(); it_x != cover_level.end(); ++it_x) {
      for (auto it_y = it_x->second.begin(); it_y != it_x->second.end(); ++it_y) {
        if (it_y->second != 2) {
          return 0;
        }
      }
    }
    return 2;
  }

  return cover_level[tile_x][tile_y];
}

bool TileManager::isCovered(unsigned int level, int tile_x, int tile_y) {
  if (level > 0) {
    if (tile_x < 0 || tile_y < 0) {
      return providesCoverage(level) == 2;
    }
    else {
      bool covered = true;
      unsigned int downsample = m_tile_information_.downsamples[level] / m_tile_information_.downsamples[level - 1];
      for (unsigned int x = 0; x < downsample; ++x) {
        for (unsigned int y = 0; y < downsample; ++y) {
          covered &= providesCoverage(level - 1, downsample * tile_x + x, downsample * tile_y + y) == 2;
        }
      }
      return covered;
    }
  }
  else {
    return false;
  }
}

void TileManager::setCoverage(unsigned int level, int tile_x, int tile_y, unsigned char covers) {
  m_tile_information_.coverage[level][tile_x][tile_y] = covers;
  if (_coverageMaps.empty()) {
    _coverageMaps.resize(m_tile_information_.last_render_level + 1);
  }
  if (level != m_tile_information_.last_render_level) {
    if (covers == 2 || covers == 0) {
      float rectSize = m_tile_information_.tile_size / (m_tile_information_.downsamples[m_tile_information_.last_render_level] / m_tile_information_.downsamples[level]);
      QPainterPath rect;
      rect.addRect(QRectF(tile_x * rectSize - 1, tile_y * rectSize - 1, rectSize + 1, rectSize + 1));
      if (covers == 2) {
        _coverageMaps[level] = _coverageMaps[level].united(rect);
      }
      else if (covers == 0) {
        if (_coverageMapCacheMode) {
          _coverageMaps[level] = _coverageMaps[level].subtracted(rect);
        }
      }
    }
  }
  emit coverageUpdated();
}

std::vector<QPainterPath> TileManager::getCoverageMaps() {
  return _coverageMaps;
}

void TileManager::clear() {
  _renderThread->clearJobs();
  while (_renderThread->getWaitingThreads() != _renderThread->getWorkers().size()) {
  }
  QCoreApplication::processEvents();
  if (_cache) {
    _cache->clear();
  }
  QList<QGraphicsItem *> itms = _scene->items();
  for (QList<QGraphicsItem *>::iterator it = itms.begin(); it != itms.end(); ++it) {
    WSITileGraphicsItem* itm = dynamic_cast<WSITileGraphicsItem*>((*it));
    if (itm) {
      _scene->removeItem(itm);
      delete itm;
    }
  }
  m_tile_information_.coverage.clear();
  _coverageMaps.clear();
  emit coverageUpdated();
}

void TileManager::refresh() {
  clear();
  QRect FOV = m_tile_information_.last_FOV;
  QPoint topLeft = FOV.topLeft();
  QPoint bottomRight = FOV.bottomRight();
  m_tile_information_.last_FOV = QRect();
  unsigned int level = m_tile_information_.last_level;
  loadAllTilesForLevel(m_tile_information_.last_render_level);
  loadTilesForFieldOfView(QRectF(tileCoordinatesToPixelCoordinates(topLeft, level), tileCoordinatesToPixelCoordinates(bottomRight, level)), level);
}