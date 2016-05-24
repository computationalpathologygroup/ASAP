#include <memory>
#include "FilterBase.h"
#include "core/ProgressMonitor.h"

FilterBase::FilterBase() : _cancel(false), _running(false) {

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


std::weak_ptr<ProgressMonitor> FilterBase::progressMonitor() const {
  return _monitor;
}

void FilterBase::setProgressMonitor(std::shared_ptr<ProgressMonitor> monitor) {
  _monitor = monitor;
}

void FilterBase::updateProgress(float progress) {
  if (std::shared_ptr<ProgressMonitor> shared_monitor = _monitor.lock()) {
    shared_monitor->setProgress(progress);
  }
}

void FilterBase::cancel() {
  _cancelMutex.lock();
  _cancel = true;
  _cancelMutex.unlock();
}