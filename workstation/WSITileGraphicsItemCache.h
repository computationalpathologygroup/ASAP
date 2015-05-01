#ifndef WSITileGraphicsItemCache_H
#define WSITileGraphicsItemCache_H

#include "io/multiresolutionimageinterface/TileCache.h"
#include <QPointer>

class WSITileGraphicsItem;

class WSITileGraphicsItemCache : public TileCache<QPointer<WSITileGraphicsItem> > {

public :
  ~WSITileGraphicsItemCache();
  void clear();
  void refresh();
  void get(const keyType& k, QPointer<WSITileGraphicsItem>& tile, unsigned int& size);
  int set(const keyType& k, QPointer<WSITileGraphicsItem> v, unsigned int size);

protected:
  void evict();

private :

  // Structure of the cache is as follow: each entry has a position string as the key (x-y-level)
  // Each value contains ((tile, size), iterator to position in _LRU)
  std::map<keyType, std::pair<std::pair<QPointer<WSITileGraphicsItem>, unsigned int>, keyTypeList::iterator> > _cache;

};

#endif