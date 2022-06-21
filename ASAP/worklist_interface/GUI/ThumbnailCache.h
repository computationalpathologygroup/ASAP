#ifndef THMBCACHE
#define THMBCACHE

#include <QString>
#include <QImage>
#include <QFileInfo>
#include <QMutex>
#include <map>

class ThumbnailCache {
public:
	ThumbnailCache(const QString& cacheFolder);
	~ThumbnailCache();

	bool addThumbnailToCache(const QString& imageFilePath, QImage thumbnail);
	QImage getThumbnailFromCache(const QString& imageFilePath);

private:
	QMutex _mutex;
	QString _cacheFolder;
	QList<QString> _cacheHashes;
	std::map<QString, QFileInfo> _hashToFileInfo;
	QFileInfoList _fileInfoLists;
	unsigned int _nrCachedFiles;
	static const unsigned int _maxCachedFiles;
};

#endif