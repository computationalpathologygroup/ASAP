#ifndef PROGRESSMONITOR_H
#define PROGRESSMONITOR_H

#include <string>
#include "core_export.h"

class CORE_EXPORT ProgressMonitor {

protected:
  std::string _status;
  unsigned int _progress;
  unsigned int _maxProgress;

public:

  ProgressMonitor();
  virtual ~ProgressMonitor();
  
  virtual void setProgress(const unsigned int& progress);
  virtual void setStatus(const std::string& status);

  std::string status() const;
  unsigned int progress() const;

  unsigned int operator++();
  unsigned int operator+=(unsigned int increment);

  virtual void setMaximumProgress(const unsigned int& maxProgress);

  unsigned int maximumProgress() const;

};

#endif