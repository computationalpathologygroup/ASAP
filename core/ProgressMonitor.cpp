#include "ProgressMonitor.h"
#include <iostream>

ProgressMonitor::ProgressMonitor() : _status(""), _progress(0), _maxProgress(100)
{

}

ProgressMonitor::~ProgressMonitor() {

}

void ProgressMonitor::setProgress(const unsigned int& progress) {
  if (progress > _maxProgress) {
    _progress = _maxProgress;
  }
  else {
    _progress = progress;
  }  
}

void ProgressMonitor::setStatus(const std::string& status) {
  _status = status; 
}

unsigned int ProgressMonitor::operator++() {
  setProgress(_progress + 1);
  return _progress;
}
unsigned int ProgressMonitor::operator += (unsigned int increment) {
  setProgress(_progress + increment);
  return _progress;
}

std::string ProgressMonitor::status() const {
  return _status;
}

unsigned int ProgressMonitor::progress() const {
  return _progress;
}

void ProgressMonitor::setMaximumProgress(const unsigned int& maxProgress) {
  _maxProgress = maxProgress;
}

unsigned int ProgressMonitor::maximumProgress() const {
  return _maxProgress;
}