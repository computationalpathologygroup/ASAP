#include "IOThread.h" 
#include "IOWorker.h" 

#include <limits>

#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "interfaces/interfaces.h"
#include "WSITileGraphicsItem.h"

using namespace pathology;

IOThread::IOThread(QObject *parent, unsigned int nrThreads) :
  QObject(parent),
  _abort(false),
  _threadsWaiting(0)
{
  for (int i = 0; i < nrThreads; ++i) {
    IOWorker* worker = new IOWorker(this);
    worker->start(QThread::HighPriority);
    _workers.push_back(worker);
  }
}

IOThread::~IOThread() 
{
  shutdown();
}

void IOThread::shutdown() {
  _abort = true;
  for (std::vector<IOWorker*>::iterator it = _workers.begin(); it != _workers.end(); ++it) {
    (*it)->abort();
    while ((*it)->isRunning()) {
      _condition.wakeOne();
    }
    delete (*it);
  }
  _workers.clear();
}

void IOThread::onBackgroundChannelChanged(int channel) {
  _jobListMutex.lock();
  for (unsigned int i = 0; i < _workers.size(); ++i) {
    _workers[i]->setBackgroundChannel(channel);
  }
  _jobListMutex.unlock();
}

void IOThread::onForegroundChannelChanged(int channel) {
  _jobListMutex.lock();
  for (unsigned int i = 0; i < _workers.size(); ++i) {
    _workers[i]->setForegroundChannel(channel);
  }
  _jobListMutex.unlock();
}

void IOThread::onLUTChanged(const pathology::LUT& LUT) {
  _jobListMutex.lock();
  for (unsigned int i = 0; i < _workers.size(); ++i) {
    _workers[i]->setLUT(LUT);
  }
  _jobListMutex.unlock();
}

std::vector<IOWorker*> IOThread::getWorkers() {
  return _workers;
}

void IOThread::addJob(const unsigned int tileSize, const long long imgPosX, const long long imgPosY, const unsigned int level, ImageSource* foregroundTile) 
{
  ThreadJob* job = NULL;
  if (foregroundTile) {
    job = new RenderJob(tileSize, imgPosX, imgPosY, level, foregroundTile);
  }
  else {
    job = new IOJob(tileSize, imgPosX, imgPosY, level);
  }
  QMutexLocker locker(&_jobListMutex);
  _jobList.push_front(job);
  _condition.wakeOne();
}

void IOThread::setForegroundImage(std::weak_ptr<MultiResolutionImage> for_img, float scale) {
  QMutexLocker locker(&_jobListMutex);
  _for_img = for_img;
  for (unsigned int i = 0; i < _workers.size(); ++i) {
    _workers[i]->setForegroundImage(for_img, scale);
  }
}

void IOThread::setBackgroundImage(std::weak_ptr<MultiResolutionImage> bck_img) {
  QMutexLocker locker(&_jobListMutex);
  _bck_img = bck_img;
  for (unsigned int i = 0; i < _workers.size(); ++i) {
    _workers[i]->setBackgroundImage(_bck_img);
  }
}

unsigned int IOThread::getWaitingThreads() {
  return _threadsWaiting;
}

ThreadJob* IOThread::getJob() {
  _jobListMutex.lock();
  while (_jobList.empty() && !_abort) {
    _threadsWaiting++;
    _condition.wait(&_jobListMutex);
    _threadsWaiting--;
  }
  if (_abort) {
    _jobListMutex.unlock();
    return NULL;
  }
  ThreadJob* job = _jobList.front();
  _jobList.pop_front();
  _jobListMutex.unlock();
  return job;
}

void IOThread::clearJobs() {
  QMutexLocker locker(&_jobListMutex);
  for (auto job : _jobList) {
    if (_workers.size() > 0) {
      if (dynamic_cast<IOJob*>(job)) {
        emit _workers[0]->tileLoaded(nullptr, job->_imgPosX, job->_imgPosY, job->_tileSize, 0, job->_level, nullptr, nullptr);
      }
      else {
        emit _workers[0]->foregroundTileRendered(nullptr, job->_imgPosX, job->_imgPosY, job->_level);
      }
    }
    delete job;
  }
  _jobList.clear();
}

unsigned int IOThread::numberOfJobs() {
  _jobListMutex.lock();
  unsigned int nrJobs = _jobList.size();
  _jobListMutex.unlock();
  return nrJobs;
}