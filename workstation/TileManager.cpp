#include "TileManager.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "RenderThread.h"
#include "WSITileGraphicsItem.h"
#include "WSITileGraphicsItemCache.h"
#include <QGraphicsScene>

TileManager::TileManager(MultiResolutionImage* img, unsigned int tileSize, RenderThread* renderThread, WSITileGraphicsItemCache* cache, QGraphicsScene* scene) :
_img(img),
_renderThread(renderThread),
_tileSize(tileSize),
_lastFOV(),
_lastLevel(),
_coverage(),
_cache(cache),
_scene(scene)
{

}

TileManager::~TileManager() {
  _renderThread = NULL;
  _img = NULL;
  _cache = NULL;
  _scene = NULL;
}

void TileManager::resetCoverage(unsigned int level) {
  _coverage[level] = std::map<int, std::map<int, bool> >();
}

QPoint TileManager::pixelCoordinatesToTileCoordinates(QPointF coordinate, unsigned int level) {
  return QPoint(std::floor((coordinate.x() / _img->getLevelDownsample(level)) / this->_tileSize), std::floor((coordinate.y() / _img->getLevelDownsample(level)) / this->_tileSize));
}

void TileManager::loadTilesForFieldOfView(const QRectF& FOV, const unsigned int level, const unsigned int channel) {
  if (_img && _renderThread) {
    QPoint topLeftTile = this->pixelCoordinatesToTileCoordinates(FOV.topLeft(), level);
    QPoint bottomRightTile = this->pixelCoordinatesToTileCoordinates(FOV.bottomRight(), level);
    QRect FOVTile = QRect(topLeftTile, bottomRightTile);
    float levelDownsample = _img->getLevelDownsample(level);
    if (FOVTile != _lastFOV || level != _lastLevel) {
      _lastLevel = level;
      _lastFOV = FOVTile;
      for (int x = topLeftTile.x(); x <= bottomRightTile.x(); ++x) {
        if (x >= 0) {
          for (int y = topLeftTile.y(); y <= bottomRightTile.y(); ++y) {
            if (y >= 0) {
              if (!providesCoverage(level, x, y)) {
                _renderThread->addJob(_tileSize, x, y, level);
              }
            }
          }
        }
      }
    }
  }
}

void TileManager::onTileLoaded(QPixmap* tile, unsigned int tileX, unsigned int tileY, unsigned int tileSize, unsigned int tileByteSize, unsigned int tileLevel, unsigned int lastRenderLevel) {
  WSITileGraphicsItem* item = new WSITileGraphicsItem(tile, tileX, tileY, tileSize, tileByteSize, tileLevel, lastRenderLevel, _img, this);
  std::stringstream ss;
  ss << tileX << "_" << tileY << "_" << tileLevel;
  std::string key;
  ss >> key;
  setCoverage(tileLevel, tileX, tileY, true);
  if (_scene) {
    float tileDownsample = _img->getLevelDownsample(tileLevel);
    float maxDownsample = _img->getLevelDownsample(lastRenderLevel);
    float posX = (tileX * tileDownsample * tileSize) / maxDownsample + ((tileSize * tileDownsample) / (2 * maxDownsample));
    float posY = (tileY * tileDownsample * tileSize) / maxDownsample + ((tileSize * tileDownsample) / (2 * maxDownsample));
    item->setPos(posX, posY);
    item->setZValue(1. / ((float)tileLevel));
    _scene->addItem(item);
  }
  if (_cache) {
    _cache->set(key, item, tileByteSize);
  }
}

bool TileManager::providesCoverage(unsigned int level, int tile_x, int tile_y) {
  std::map<int, std::map<int, bool> >& cover_level = _coverage[level];
  if (cover_level.empty()) {
    return false;
  }

  if (tile_x < 0 || tile_y < 0) {
    for (std::map<int, std::map<int, bool> >::iterator it_x = cover_level.begin(); it_x != cover_level.end(); ++it_x) {
      for (std::map<int, bool>::iterator it_y = it_x->second.begin(); it_y != it_x->second.end(); ++it_y) {
        if (!it_y->second) {
          return false;
        }
      }
    }
    return true;
  }

  return cover_level[tile_x][tile_y];
}

bool TileManager::isCovered(unsigned int level, int tile_x, int tile_y) {
  if (tile_x < 0 || tile_y < 0) {
    return providesCoverage(level);
  }
  else {
    return (providesCoverage(level - 1, 2 * tile_x, 2 * tile_y) &&
            providesCoverage(level - 1, 2 * tile_x, 2 * tile_y + 1) &&
            providesCoverage(level - 1, 2 * tile_x + 1, 2 * tile_y) &&
            providesCoverage(level - 1, 2 * tile_x + 1, 2 * tile_y + 1));
  }
}

void TileManager::setCoverage(unsigned int level, int tile_x, int tile_y, bool covers) {
  _coverage[level][tile_x][tile_y] = covers;
}