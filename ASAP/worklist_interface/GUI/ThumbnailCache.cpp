#include "ThumbnailCache.h"
#include <QDir>
#include <QCryptographicHash>
#include <QFileInfoList>
#include <QFile>
#include <QDateTime>

const unsigned int ThumbnailCache::_maxCachedFiles = 1000;

ThumbnailCache::ThumbnailCache(const QString& cacheFolder) : _cacheFolder(cacheFolder + "/thumbnails"), _nrCachedFiles(0)
{
	QDir cfolder(_cacheFolder);
	if (cfolder.exists()) {
		_fileInfoLists = cfolder.entryInfoList(QDir::Files, QDir::Time);
		_nrCachedFiles = _fileInfoLists.size();
		for (auto fileInfo : _fileInfoLists) {
			QString hash = fileInfo.completeBaseName();
			_hashToFileInfo.insert({ hash, fileInfo });
		}
	}
	else {
		QDir().mkpath(_cacheFolder);
	}

	// BUG: this is a workaround for a bug in QDateTime.toString() not being reentrant or thread-safe: https://bugreports.qt.io/browse/QTBUG-85692
	QDateTime workaroundBug = QDateTime::currentDateTime();
	workaroundBug.toString(Qt::ISODate);
}

ThumbnailCache::~ThumbnailCache()
{

}

bool ThumbnailCache::addThumbnailToCache(const QString& imageFilePath, QImage thumbnail)
{
	QFileInfo img_info(imageFilePath);
	thumbnail.setText("imgFilePath", imageFilePath);
	QString fileHash = QCryptographicHash::hash((imageFilePath + img_info.lastModified().toString(Qt::ISODate)).toUtf8(), QCryptographicHash::Algorithm::Md5).toHex();
	while (_nrCachedFiles >= _maxCachedFiles) {
		QFileInfo thumbnail = _fileInfoLists.first();
		bool succes = QFile::remove(thumbnail.canonicalFilePath());
		if (succes) {
			_mutex.lock();
			_fileInfoLists.pop_front();
			_nrCachedFiles--;
			_mutex.unlock();
		}
		else {
			break;
		}
	}
	QString thumbnailFilePath = _cacheFolder + "/" + fileHash + ".png";
	thumbnail.save(thumbnailFilePath);
	QFileInfo thumbInfo(thumbnailFilePath);
	_mutex.lock();
	_fileInfoLists.push_back(thumbInfo);
	_hashToFileInfo.insert({ fileHash, thumbInfo });
	_nrCachedFiles++;	
	_mutex.unlock();
	return true;
}



QImage ThumbnailCache::getThumbnailFromCache(const QString& imageFilePath)
{
	QFileInfo img_info(imageFilePath);
	QString file_hash = QCryptographicHash::hash((imageFilePath + img_info.lastModified().toString(Qt::ISODate)).toUtf8(), QCryptographicHash::Algorithm::Md5).toHex();
	QString thumbnailFilePath = _cacheFolder + "/" + file_hash + ".png";
	if (_hashToFileInfo.find(file_hash) != _hashToFileInfo.end()) {
		QImage thumbnail(thumbnailFilePath);
		return thumbnail;
	}
	else {
		return QImage();
	}
}
