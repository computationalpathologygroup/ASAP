#include "WSITileGraphicsItemCache.h"
#include "WSITileGraphicsItem.h"

WSITileGraphicsItemCache::~WSITileGraphicsItemCache(void)
{
	m_mutex_.lock();
	m_mutex_.unlock();
}

void WSITileGraphicsItemCache::clear(void)
{
	_cache.clear();
	_LRU.clear();
	_cacheCurrentByteSize = 0;
}

void WSITileGraphicsItemCache::get(const keyType& k, WSITileGraphicsItem* tile, uint32_t& size)
{
	m_mutex_.lock();
	auto it = _cache.find(k);
	if (it == _cache.end())
	{
		tile = nullptr;
	}
	else 
	{
		if (it->second.second != _LRU.end())
		{
			_LRU.splice(_LRU.end(), _LRU, it->second.second);
		}
		tile = it->second.first.first;
		size = it->second.first.second;
	}
	m_mutex_.unlock();
}

bool WSITileGraphicsItemCache::set(const keyType& k, WSITileGraphicsItem* v, uint32_t size, bool topLevel)
{
	m_mutex_.lock();
	if (_cache.find(k) != _cache.end() ||
		size > _cacheMaxByteSize)
	{
		m_mutex_.unlock();
		return false;
	}

	while (_cacheCurrentByteSize + size > _cacheMaxByteSize && _cacheCurrentByteSize != 0)
	{
		evict();
	}

	// Do not add to the LRU if it is a top-level item so it is never removed
	if (!topLevel)
	{
		keyTypeList::iterator it = _LRU.insert(_LRU.end(), k);
		_cache[k] = std::make_pair(std::make_pair(v, size), it);
	}
	else
	{
		_cache[k] = std::make_pair(std::make_pair(v, size), _LRU.end());
	}
	_cacheCurrentByteSize += size;
	m_mutex_.unlock();
	return true;
}

void WSITileGraphicsItemCache::evict(void)
{
	// Identify least recently used key 
	auto it = _cache.find(_LRU.front());

	// Erase both elements to completely purge record 
	WSITileGraphicsItem* itemToEvict = it->second.first.first;
	_cacheCurrentByteSize -= it->second.first.second;
	_cache.erase(it);
	_LRU.pop_front();
	emit itemEvicted(itemToEvict);
	delete itemToEvict;
}