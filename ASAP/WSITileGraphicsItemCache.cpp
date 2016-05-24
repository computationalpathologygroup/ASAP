#include "WSITileGraphicsItemCache.h"
#include "WSITileGraphicsItem.h"

WSITileGraphicsItemCache::~WSITileGraphicsItemCache() {
  clear();
}

void WSITileGraphicsItemCache::evict() {
  // Identify least recently used key 
  std::map<keyType, std::pair<std::pair<WSITileGraphicsItem*, unsigned int>, keyTypeList::iterator> >::iterator it = _cache.find(_LRU.front());

  // Erase both elements to completely purge record 
  WSITileGraphicsItem* itemToEvict = it->second.first.first;
  _cacheCurrentByteSize -= it->second.first.second;
  _cache.erase(it);
  _LRU.pop_front();
  emit itemEvicted(itemToEvict);
}

void WSITileGraphicsItemCache::clear() {
  _cache.clear();
  _LRU.clear();
  _cacheCurrentByteSize = 0;
}

void WSITileGraphicsItemCache::get(const keyType& k, WSITileGraphicsItem* tile, unsigned int& size) {

  std::map<keyType, std::pair<std::pair<WSITileGraphicsItem*, unsigned int>, keyTypeList::iterator> >::iterator it = _cache.find(k);

  if (it == _cache.end()) {
    tile = NULL;
    return;
  }
  else {
    if (it->second.second != _LRU.end()) {
      _LRU.splice(
        _LRU.end(),
        _LRU,
        it->second.second
        );
    }
    tile = it->second.first.first;
    size = it->second.first.second;
    return;
  }
}

int WSITileGraphicsItemCache::set(const keyType& k, WSITileGraphicsItem* v, unsigned int size, bool topLevel) {
  if (_cache.find(k) != _cache.end()) {
    return 1;
  }
  if (size > _cacheMaxByteSize) {
    return 1;
  }
  while (_cacheCurrentByteSize + size > _cacheMaxByteSize && _cacheCurrentByteSize != 0) {
    evict();
  }

  // Do not add to the LRU if it is a top-level item so it is never removed
  if (!topLevel) {
    keyTypeList::iterator it = _LRU.insert(_LRU.end(), k);
    _cache[k] = std::make_pair(std::make_pair(v, size), it);
  }
  else {
    _cache[k] = std::make_pair(std::make_pair(v, size), _LRU.end());
  }
  _cacheCurrentByteSize += size;
  return 0;
}