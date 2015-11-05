#ifndef TILECACHE_H
#define TILECACHE_H
#include <list>
#include <map>
#include <string>

template <typename T>
class TileCache {
public :

  TileCache() :
    _cacheMaxByteSize(0),
    _cacheCurrentByteSize(0),
    _LRU(),
    _cache()
  { 
  }

  TileCache(unsigned int cacheMaxByteSize) :
    _cacheMaxByteSize(cacheMaxByteSize),
    _cacheCurrentByteSize(0),
    _LRU(),
    _cache()
  { 
  }

  virtual ~TileCache()
  {
    for (key_iterator it = _cache.begin(); it != _cache.end(); ++it) {
      delete[] it->second.first.first;
    }
  }

  typedef std::string keyType;
  typedef std::list<keyType> keyTypeList;
  typedef typename std::map<keyType,std::pair<std::pair<T*, unsigned int>,std::list<keyType>::iterator> >::iterator key_iterator;  

  virtual void get(const keyType& k, T*& tile, unsigned int& size) {
    key_iterator it  = _cache.find(k);
 
    if (it == _cache.end()) {  
      tile = NULL;
      return;
    } else { 
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

  virtual int set(const keyType& k, T* v, unsigned int size) {
    if (_cache.find(k) != _cache.end()) {
      return 1;
    }
    if (size > _cacheMaxByteSize) {
      return 1;
    }
    while (_cacheCurrentByteSize + size > _cacheMaxByteSize && _cacheCurrentByteSize != 0) {
      evict();
    }

    keyTypeList::iterator it =_LRU.insert(_LRU.end(),k); 
    _cache[k] = std::make_pair(std::make_pair(v,size),it);
    _cacheCurrentByteSize += size;
    return 0;
  }

  unsigned long long currentCacheSize() const {return _cacheCurrentByteSize;}
  unsigned long long maxCacheSize() const {return _cacheMaxByteSize;}
  void setMaxCacheSize(const unsigned long long& cacheMaxByteSize) {
    _cacheMaxByteSize = cacheMaxByteSize;
    while (_cacheCurrentByteSize > _cacheMaxByteSize) {
      evict();
    }
  }

  virtual void clear() {
    for (key_iterator it = _cache.begin(); it != _cache.end(); ++it) {
      delete[] it->second.first.first;
    }
    _cache.clear();
    _LRU.clear();
    _cacheCurrentByteSize = 0;
  }

protected :
  unsigned long long _cacheCurrentByteSize;
  unsigned long long _cacheMaxByteSize;

  // _LRU is used to quickly identify the last used tiles so they can be removed from the cache,
  // it contains an ordered list of used tiles. When a tile is used it is moved to the front of the 
  // list
  std::list<keyType> _LRU;

  // Structure of the cache is as follow: each entry has a position string as the key (x-y-level)
  // Each value contains ((tile, size), iterator to position in _LRU)
  std::map<keyType,std::pair<std::pair<T*, unsigned int>,keyTypeList::iterator> > _cache;


  // Removes the least recently used (LRU) tile from the cache
  virtual void evict() {

    // Identify least recently used key 
    key_iterator it =_cache.find(_LRU.front()); 

    // Erase both elements to completely purge record 
    _cacheCurrentByteSize -= it->second.first.second;
    delete[] it->second.first.first;
    _cache.erase(it); 
    _LRU.pop_front(); 

  }

};

#endif