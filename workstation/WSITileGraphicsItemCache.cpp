#include "WSITileGraphicsItemCache.h"
#include "WSITileGraphicsItem.h"

WSITileGraphicsItemCache::~WSITileGraphicsItemCache() {
  for (std::map<keyType, std::pair<std::pair<QPointer<WSITileGraphicsItem>, unsigned int>, std::list<keyType>::iterator> >::iterator it = _cache.begin(); it != _cache.end(); ++it) {
    if (it->second.first.first) {
      it->second.first.first->deleteLater();
    }
  }
}

void WSITileGraphicsItemCache::evict() {
  // Identify least recently used key 
  std::map<keyType, std::pair<std::pair<QPointer<WSITileGraphicsItem>, unsigned int>, keyTypeList::iterator> >::iterator it = _cache.find(_LRU.front());

  // Erase both elements to completely purge record 
  _cacheCurrentByteSize -= it->second.first.second;
  if (it->second.first.first) {
    it->second.first.first->deleteLater();
  }
  _cache.erase(it);
  _LRU.pop_front();
}

void WSITileGraphicsItemCache::refresh() {
  for (std::map<keyType, std::pair<std::pair<QPointer<WSITileGraphicsItem>, unsigned int>, std::list<keyType>::iterator> >::iterator it = _cache.begin(); it != _cache.end(); ++it) {
    if (it->second.first.first) {
      it->second.first.first->refreshItem();
    }
  }
}

void WSITileGraphicsItemCache::clear() {
  for (std::map<keyType, std::pair<std::pair<QPointer<WSITileGraphicsItem>, unsigned int>, std::list<keyType>::iterator> >::iterator it = _cache.begin(); it != _cache.end(); ++it) {
    if (it->second.first.first) {
      it->second.first.first->deleteLater();
    }
  }
  _cache.clear();
  _LRU.clear();
  _cacheCurrentByteSize = 0;
}

void WSITileGraphicsItemCache::get(const keyType& k, QPointer<WSITileGraphicsItem>& tile, unsigned int& size) {

  std::map<keyType, std::pair<std::pair<QPointer<WSITileGraphicsItem>, unsigned int>, keyTypeList::iterator> >::iterator it = _cache.find(k);

  if (it == _cache.end()) {
    tile = NULL;
    return;
  }
  else {
    _LRU.splice(
      _LRU.end(),
      _LRU,
      it->second.second
      );
    tile = it->second.first.first;
    size = it->second.first.second;
    return;
  }
}

int WSITileGraphicsItemCache::set(const keyType& k, QPointer<WSITileGraphicsItem> v, unsigned int size) {
  if (_cache.find(k) != _cache.end()) {
    return 1;
  }
  if (size > _cacheMaxByteSize) {
    return 1;
  }
  while (_cacheCurrentByteSize + size > _cacheMaxByteSize && _cacheCurrentByteSize != 0) {
    evict();
  }

  keyTypeList::iterator it = _LRU.insert(_LRU.end(), k);
  _cache[k] = std::make_pair(std::make_pair(v, size), it);
  _cacheCurrentByteSize += size;
  return 0;
}