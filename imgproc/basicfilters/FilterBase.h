#ifndef _FilterBase
#define _FilterBase

#include "basicfilters_export.h"
#include <mutex>
#include <string>
#include <memory>

class ProgressMonitor;

class BASICFILTERS_EXPORT FilterBase {

  std::mutex _cancelMutex;
  std::weak_ptr<ProgressMonitor> _monitor;
  bool _cancel;
  bool _running;

public:

  FilterBase();
  FilterBase(const FilterBase& other);
  FilterBase& operator=(const FilterBase& rhs);
  virtual ~FilterBase();

  virtual std::string name() const;

  std::weak_ptr<ProgressMonitor> progressMonitor() const;
  void setProgressMonitor(std::shared_ptr<ProgressMonitor> monitor);
  void updateProgress(float progress);

  void cancel();
  void start();
  void finish();
  bool shouldCancel() {
    if (_running && _cancel) {
      finish();
      return true;
    }
    return false;
  }

};

#endif 