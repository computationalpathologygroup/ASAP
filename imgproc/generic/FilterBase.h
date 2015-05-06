#ifndef _FilterBase
#define _FilterBase

#include "config/pathology_config.h"
#include <mutex>
#include <string>

class ProgressMonitor;

class EXPORT_BASICFILTERS FilterBase {

  std::mutex _monitorMutex;
  std::mutex _cancelMutex;
  ProgressMonitor* _monitor;
  bool _cancel;
  bool _running;

public:

  FilterBase();
  FilterBase(const FilterBase& other);
  FilterBase& operator=(const FilterBase& rhs);
  virtual ~FilterBase();

  virtual std::string name() const;

  const ProgressMonitor* const progressMonitor() const;
  void setProgressMonitor(ProgressMonitor* monitor);
  void updateProgress(float progress);

  void cancel();
  void start();
  void finish();
  inline bool shouldCancel() {
    if (_running && _cancel) {
      finish();
      return true;
    }
    return false;
  }

};

#endif 