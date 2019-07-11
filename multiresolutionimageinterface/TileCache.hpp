#ifndef TILECACHE_H
#define TILECACHE_H
#include <list>
#include <map>
#include <string>

template <typename T>
class TileCache
{
	public:
		typedef std::string keyType;
		typedef std::list<keyType> keyTypeList;
		typedef typename std::map<keyType, std::pair<std::pair<T*, uint32_t>, std::list<keyType>::iterator>>::iterator key_iterator;

		TileCache() :_cacheMaxByteSize(0), _cacheCurrentByteSize(0), _LRU(), _cache()
		{
		}

		TileCache(uint64_t cacheMaxByteSize) : _cacheMaxByteSize(cacheMaxByteSize), _cacheCurrentByteSize(0), _LRU(), _cache()
		{
		}

		~TileCache(void)
		{
			clear();
		}

		virtual void get(const keyType& k, T*& tile, uint32_t& size)
		{
			key_iterator it = _cache.find(k);

			if (it == _cache.end())
			{
				tile = nullptr;
			}
			else
			{
				_LRU.splice(_LRU.end(), _LRU, it->second.second);
				tile = it->second.first.first;
				size = it->second.first.second;
			}
		}

		virtual bool set(const keyType& k, T* v, const uint32_t size)
		{
			if (_cache.find(k) != _cache.end() ||
				size > _cacheMaxByteSize)
			{
				return false;
			}

			while (_cacheCurrentByteSize + size > _cacheMaxByteSize && _cacheCurrentByteSize != 0)
			{
				evict();
			}

			keyTypeList::iterator it =_LRU.insert(_LRU.end(),k); 
			_cache[k] = std::make_pair(std::make_pair(v,size),it);
			_cacheCurrentByteSize += size;
			return true;
		}

		uint64_t getCurrentCacheSize(void) const
		{
			return _cacheCurrentByteSize;
		}

		uint64_t getMaxCacheSize(void) const
		{
			return _cacheMaxByteSize;
		}

		void setMaxCacheSize(const uint64_t cacheMaxByteSize)
		{
			_cacheMaxByteSize = cacheMaxByteSize;
			while (_cacheCurrentByteSize > _cacheMaxByteSize)
			{
				evict();
			}
		}

		virtual void clear(void)
		{
			for (key_iterator it = _cache.begin(); it != _cache.end(); ++it)
			{
				delete[] it->second.first.first;
			}

			_cache.clear();
			_LRU.clear();
			_cacheCurrentByteSize = 0;
		}

	protected :
		uint64_t _cacheCurrentByteSize;
		uint64_t _cacheMaxByteSize;

		// _LRU is used to quickly identify the last used tiles so they can be removed from the cache,
		// it contains an ordered list of used tiles. When a tile is used it is moved to the front of the 
		// list
		std::list<keyType> _LRU;

		// Structure of the cache is as follow: each entry has a position string as the key (x-y-level)
		// Each value contains ((tile, size), iterator to position in _LRU)
		std::map<keyType,std::pair<std::pair<T*, uint32_t>,keyTypeList::iterator>> _cache;

		// Removes the least recently used (LRU) tile from the cache
		virtual void evict(void)
		{
			// Identify least recently used key 
			key_iterator it =_cache.find(_LRU.front()); 

			// Erase both elements to completely purge record 
			_cacheCurrentByteSize -= it->second.first.second;
			delete[] it->second.first.first;
			_cache.erase(it); 
			_LRU.pop_front(); 
		}
};
#endif // TILECACHE_H