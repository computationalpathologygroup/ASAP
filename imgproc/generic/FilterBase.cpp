#include "FilterBase.h"
#include "core/ProgressMonitor.h"

FilterBase::FilterBase() : _monitor(NULL), _cancel(false), _running(false) {

}

FilterBase::~FilterBase()
{
};

FilterBase::FilterBase(const FilterBase& other) : _monitor(other._monitor), _cancel(other._cancel) {
}

FilterBase& FilterBase::operator=(const FilterBase& rhs) {
  this->_monitor = rhs._monitor;
  this->_cancel = rhs._cancel;
  return *this;
}

std::string FilterBase::name() const {
  return std::string("FilterBase");
};

void FilterBase::start() {
  if (!_running) {
    _cancelMutex.lock();
    _running = true;
    _cancel = false;
    _cancelMutex.unlock();
  }
}

void FilterBase::finish() {
  if (_running) {
    _cancelMutex.lock();
    _cancel = false;
    _running = false;
    _cancelMutex.unlock();
  }
}


const ProgressMonitor* const FilterBase::progressMonitor() const {
  return _monitor;
}

void FilterBase::setProgressMonitor(ProgressMonitor* monitor) {
  _monitorMutex.lock();
  _monitor = monitor;
  _monitorMutex.unlock();
}

void FilterBase::updateProgress(float progress) {
  _monitorMutex.lock();
  if (_monitor) {
    _monitor->setProgress(progress);
  }
  _monitorMutex.unlock();
}

void FilterBase::cancel() {
  _cancelMutex.lock();
  _cancel = true;
  _cancelMutex.unlock();
}