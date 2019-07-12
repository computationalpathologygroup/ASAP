#ifndef WSITileGraphicsItemCache_H
#define WSITileGraphicsItemCache_H

#include <mutex>
#include "multiresolutionimageinterface/TileCache.hpp"
#include <QObject>

class WSITileGraphicsItem;
class WSITileGraphicsItemCache : public QObject, public TileCache<WSITileGraphicsItem> 
{
	Q_OBJECT
	public:
		~WSITileGraphicsItemCache(void);

		void clear(void);
		void get(const keyType& k, WSITileGraphicsItem* tile, uint32_t& size);
		bool set(const keyType& k, WSITileGraphicsItem* v, uint32_t size, bool topLevel = false);

	protected:
		void evict(void);

	private:
		std::mutex m_mutex_;

	signals:
		void itemEvicted(WSITileGraphicsItem* item);
};
#endif // WSITileGraphicsItemCache_H