#include "QtProgressMonitor.h"

void QtProgressMonitor::setProgress(const unsigned int& progress) {
  ProgressMonitor::setProgress(progress);
  emit progressChanged(100 * static_cast<float>(_progress) / static_cast<float>(_maxProgress));
}

void QtProgressMonitor::setStatus(const std::string& status) {
  ProgressMonitor::setStatus(status);
  emit statusChanged(status);
}