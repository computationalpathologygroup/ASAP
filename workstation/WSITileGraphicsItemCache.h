#ifndef WSITileGraphicsItemCache_H
#define WSITileGraphicsItemCache_H

#include "io/multiresolutionimageinterface/TileCache.h"
#include <QObject>

class WSITileGraphicsItem;

class WSITileGraphicsItemCache : public QObject, public TileCache<WSITileGraphicsItem* >  {
  Q_OBJECT

public :
  ~WSITileGraphicsItemCache();
  void clear();
  void get(const keyType& k, WSITileGraphicsItem* tile, unsigned int& size);
  int set(const keyType& k, WSITileGraphicsItem* v, unsigned int size, bool topLevel = false);

protected:
  void evict();

private :

  // Structure of the cache is as follow: each entry has a position string as the key (x-y-level)
  // Each value contains ((tile, size), iterator to position in _LRU)
  std::map<keyType, std::pair<std::pair<WSITileGraphicsItem*, unsigned int>, keyTypeList::iterator> > _cache;

signals:
  void itemEvicted(WSITileGraphicsItem* item);
};

#endif