#include "PrefetchThread.h" 

#include <QPixmap>
#include <QElapsedTimer>
#include <QDebug>

#include "multiresolutionimageinterface/MultiResolutionImage.h"

PrefetchThread::PrefetchThread(QObject *parent) :
  QThread(parent),
  _restart(false),
  _abort(false),
  _FOV(QRectF()),
  _level(0),
  _img(NULL)
{
}

PrefetchThread::~PrefetchThread() 
{
  _mutex.lock();
  _abort = true;
  _mutex.unlock();
  while (isRunning()) {
    _condition.wakeOne();
  }

  wait();
}

void PrefetchThread::FOVChanged(MultiResolutionImage* img, const QRectF& FOV, const unsigned int level)
{
  QMutexLocker locker(&_mutex);

  _img = img;
  _level = level;
  _FOV = FOV;

  if (!isRunning()) {
      start(HighPriority);
  } else {
      _restart = true;
      _condition.wakeOne();
  }
}

// Fetch threads based on the current FOV and level; prefetch will cache one level up and one level down of the
// current FOV (taking into account the downsampling) and tiles outside the FOV with the same size of the FOV in
// a 8 connected fashion
void PrefetchThread::run() 
{
  forever{
    if (_img) {
      _mutex.lock();
      QRectF FOV = _FOV;
      int level = _level;
      MultiResolutionImage* img = _img;
      std::vector<unsigned long long> L0Dims = img->getDimensions();
      QRectF maxFOV(0, 0, L0Dims[0], L0Dims[1]);
      int levelDownsample = img->getLevelDownsample(level);
      _mutex.unlock();

      if (img) {
        if (_abort) {
          return;
        }
        // Cache level below
        if (level - 1 >= 0 && !_restart) {
          unsigned int width = FOV.width() / img->getLevelDownsample(level - 1);
          unsigned int height = FOV.height() / img->getLevelDownsample(level - 1);
          unsigned char* data = NULL;
          data = new unsigned char[width*height*img->getSamplesPerPixel()];
          img->getRawRegion(FOV.left(), FOV.top(), width, height, level - 1, data);
          delete[] data;
        }
        // Cache around current FOV if needed
        for (int x = -1; x < 2; ++x) {
          if (_restart) {
            break;
          }
          for (int y = -1; y < 2; ++y) {
            if (_restart) {
              break;
            }
            if (x == 0 && y == 0) {
              continue;
            }
            QRectF cur(FOV.left() + x*FOV.width(), FOV.top() + y*FOV.height(), FOV.width(), FOV.height());
            cur = cur.intersected(maxFOV);
            unsigned int width = cur.width() / levelDownsample;
            unsigned int height = cur.height() / levelDownsample;
            unsigned char* data = new unsigned char[width*height*img->getSamplesPerPixel()];
            img->getRawRegion(cur.left(), cur.top(), width, height, level, data);
            delete[] data;
          }
        }
      }

      _mutex.lock();
      if (!_restart) {
        qDebug() << "Prefetching finished!";
        _condition.wait(&_mutex);
      }
      _restart = false;
      _mutex.unlock();
      if (_abort) {
        return;
      }
    }
  }
}