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

TileManager::TileManager(std::shared_ptr<MultiResolutionImage> img, unsigned int tileSize, unsigned int lastRenderLevel, RenderThread* renderThread, WSITileGraphicsItemCache* cache, QGraphicsScene* scene) :
_renderThread(renderThread),
_tileSize(tileSize),
_lastRenderLevel(lastRenderLevel),
_lastFOV(),
_lastLevel(),
_coverage(),
_cache(cache),
_scene(scene),
_coverageMaps(),
_coverageMapCacheMode(false)
{
  for (unsigned int i = 0; i < img->getNumberOfLevels(); ++i) {
    _levelDownsamples.push_back(img->getLevelDownsample(i));
    _levelDimensions.push_back(img->getLevelDimensions(i));
  }
}

TileManager::~TileManager() {
  _renderThread = NULL;
  _cache = NULL;
  _scene = NULL;
}

void TileManager::resetCoverage(unsigned int level) {
  _coverage[level] = std::map<int, std::map<int, unsigned char> >();
  if (_coverageMaps.size() > level) {
    _coverageMaps[level] = QPainterPath();
  }
}

QPoint TileManager::pixelCoordinatesToTileCoordinates(QPointF coordinate, unsigned int level) {
  if (level < _levelDownsamples.size()) {
    return QPoint(std::floor((coordinate.x() / _levelDownsamples[level]) / this->_tileSize), std::floor((coordinate.y() / _levelDownsamples[level]) / this->_tileSize));
  }
  else {
    return QPoint();
  }
}

QPointF TileManager::tileCoordinatesToPixelCoordinates(QPoint coordinate, unsigned int level) {  
  if (level < _levelDownsamples.size()) {
    return QPointF(coordinate.x() * _levelDownsamples[level] * this->_tileSize, coordinate.y() * _levelDownsamples[level] * this->_tileSize);
  }
  else {
    return QPoint();
  }
}

QPoint TileManager::getLevelTiles(unsigned int level) {
  if (level < _levelDimensions.size()) {
    std::vector<unsigned long long> dims = _levelDimensions[level];
    return QPoint(std::ceil(dims[0] / static_cast<float>(_tileSize)), std::ceil(dims[1] / static_cast<float>(_tileSize)));
  }
  else {
    return QPoint();
  }
}

void TileManager::loadAllTilesForLevel(unsigned int level) {
  if (_renderThread) {
    if (level < _levelDownsamples.size()) {
      std::vector<unsigned long long> baseLevelDims = _levelDimensions[0];
      this->loadTilesForFieldOfView(QRectF(0, 0, baseLevelDims[0], baseLevelDims[1]), level);
    }
  }
}

void TileManager::loadTilesForFieldOfView(const QRectF& FOV, const unsigned int level) {
  if (level > _lastRenderLevel) {
    return;
  }
  if (_renderThread) {
    QPoint topLeftTile = this->pixelCoordinatesToTileCoordinates(FOV.topLeft(), level);
    QPoint bottomRightTile = this->pixelCoordinatesToTileCoordinates(FOV.bottomRight(), level);
    QRect FOVTile = QRect(topLeftTile, bottomRightTile);
    QPoint nrTiles = getLevelTiles(level);
    float levelDownsample = _levelDownsamples[level];
    if (FOVTile != _lastFOV || level != _lastLevel) {
      _lastLevel = level;
      _lastFOV = FOVTile;
      for (int x = topLeftTile.x(); x <= bottomRightTile.x(); ++x) {
        if (x >= 0 && x <= nrTiles.x()) {
          for (int y = topLeftTile.y(); y <= bottomRightTile.y(); ++y) {
            if (y >= 0 && y <= nrTiles.y()) {
              if (providesCoverage(level, x, y) < 1) {
                setCoverage(level, x, y, 1);
                _renderThread->addJob(_tileSize, x, y, level);
              }
            }
          }
        }
      }
    }
  }
}

void TileManager::onTileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel) {
  WSITileGraphicsItem* item = new WSITileGraphicsItem(tile, tileX, tileY, tileSize, tileByteSize, tileLevel, _lastRenderLevel, _levelDownsamples, this);
  std::stringstream ss;
  ss << tileX << "_" << tileY << "_" << tileLevel;
  std::string key;
  ss >> key;
  if (_scene) {
    setCoverage(tileLevel, tileX, tileY, 2);
    float tileDownsample = _levelDownsamples[tileLevel];
    float maxDownsample = _levelDownsamples[_lastRenderLevel];
    float posX = (tileX * tileDownsample * tileSize) / maxDownsample + ((tileSize * tileDownsample) / (2 * maxDownsample));
    float posY = (tileY * tileDownsample * tileSize) / maxDownsample + ((tileSize * tileDownsample) / (2 * maxDownsample));
    _scene->addItem(item);
    item->setPos(posX, posY);
    item->setZValue(1. / ((float)tileLevel + 1.));
  }
  if (_cache) {
    _cache->set(key, item, tileByteSize, tileLevel == _lastRenderLevel);
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
  std::map<int, std::map<int, unsigned char> >& cover_level = _coverage[level];
  if (cover_level.empty()) {
    return 0;
  }

  if (tile_x < 0 || tile_y < 0) {
    for (std::map<int, std::map<int, unsigned char> >::iterator it_x = cover_level.begin(); it_x != cover_level.end(); ++it_x) {
      for (std::map<int, unsigned char>::iterator it_y = it_x->second.begin(); it_y != it_x->second.end(); ++it_y) {
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
      unsigned int downsample = _levelDownsamples[level] / _levelDownsamples[level - 1];
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
  _coverage[level][tile_x][tile_y] = covers;
  if (_coverageMaps.empty()) {
    _coverageMaps.resize(_lastRenderLevel + 1);
  }
  if (level != _lastRenderLevel) {
    if (covers == 2 || covers == 0) {
      float rectSize = _tileSize / (_levelDownsamples[_lastRenderLevel] / _levelDownsamples[level]);
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
  _coverage.clear();
  _coverageMaps.clear();
  emit coverageUpdated();
}

void TileManager::refresh() {
  clear();
  QRect FOV = _lastFOV;
  QPoint topLeft = FOV.topLeft();
  QPoint bottomRight = FOV.bottomRight();
  _lastFOV = QRect();
  unsigned int level = _lastLevel;
  loadAllTilesForLevel(_lastRenderLevel);
  loadTilesForFieldOfView(QRectF(tileCoordinatesToPixelCoordinates(topLeft, level), tileCoordinatesToPixelCoordinates(bottomRight, level)), level);
}