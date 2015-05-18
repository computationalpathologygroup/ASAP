#include "RenderThread.h" 
#include "RenderWorker.h" 

#include <limits>

#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "interfaces/interfaces.h"
#include "WSITileGraphicsItem.h"

using namespace pathology;

RenderThread::RenderThread(MultiResolutionImage* bck_img, MultiResolutionImage* for_img, unsigned int nrThreads, QObject *parent) :
  QObject(parent),
  _bck_img(bck_img),
  _for_img(for_img),
  _abort(false),
  _channel(0)
{
  for (int i = 0; i < nrThreads; ++i) {
    RenderWorker* worker = new RenderWorker(this, _bck_img, _for_img);
    worker->start(QThread::HighPriority);
    _workers.push_back(worker);
  }
}

RenderThread::~RenderThread() 
{
  shutdown();
}

void RenderThread::shutdown() {
  _abort = true;
  for (std::vector<RenderWorker*>::iterator it = _workers.begin(); it != _workers.end(); ++it) {
    (*it)->abort();
    while ((*it)->isRunning()) {
      _condition.wakeOne();
    }
    delete (*it);
  }
  _workers.clear();
}

void RenderThread::setForegroundOpacity(const float& opacity) {
  _jobListMutex.lock();
  _opacity = opacity;
  for (unsigned int i = 0; i < _workers.size(); ++i) {
    _workers[i]->setForegroundOpacity(_opacity);
  }
  _jobListMutex.unlock();
}

float RenderThread::getForegroundOpacity() const {
  return _opacity;
}

void RenderThread::onChannelChanged(int channel) {
  _jobListMutex.lock();
  _channel = channel;
  for (unsigned int i = 0; i < _workers.size(); ++i) {
    _workers[i]->setChannel(_channel);
  }
  _jobListMutex.unlock();
}

void RenderThread::addJob(const unsigned int tileSize, const unsigned int samplesPerPixel, const long long imgPosX, const long long imgPosY, const unsigned int level, QPointer<WSITileGraphicsItem> sender) 
{
    RenderJob job = {tileSize, samplesPerPixel, imgPosX, imgPosY, level, sender};

    QMutexLocker locker(&_jobListMutex);
    _jobList.push_front(job);
    _condition.wakeOne();
}

void RenderThread::setForegroundImage(MultiResolutionImage* for_img) {
  QMutexLocker locker(&_jobListMutex);
  _for_img = for_img;
  for (unsigned int i = 0; i < _workers.size(); ++i) {
    _workers[i]->setForegroundImage(for_img);
  }
}

RenderJob RenderThread::getJob() {
  _jobListMutex.lock();
  while (_jobList.empty() && !_abort) {
    _condition.wait(&_jobListMutex);
  }
  if (_abort) {
    _jobListMutex.unlock();
    return RenderJob();
  }
  RenderJob job = _jobList.front();
  _jobList.pop_front();
  _jobListMutex.unlock();
  return job;
}

void RenderThread::clearJobs() {
  QMutexLocker locker(&_jobListMutex);
  for (std::list<RenderJob>::iterator it = _jobList.begin(); it != _jobList.end(); ++it) {
    _senderDeletionMutex.lock();
    if (it->_sender) {
      it->_sender->setVisible(true);
    }
    _senderDeletionMutex.unlock();
  }
  _jobList.clear();
}

unsigned int RenderThread::numberOfJobs() {
  _jobListMutex.lock();
  unsigned int nrJobs = _jobList.size();
  _jobListMutex.unlock();
  return nrJobs;
}