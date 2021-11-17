#ifndef CMDLINEPROGRESSMONITOR_H
#define CMDLINEPROGRESSMONITOR_H

#include "core_export.h"
#include "ProgressMonitor.h"

class ProgressDisplay;

class CORE_EXPORT CmdLineProgressMonitor : public ProgressMonitor {

protected:
  ProgressDisplay* disp;

public :

  CmdLineProgressMonitor();
  virtual ~CmdLineProgressMonitor();

  void setProgress(const unsigned int& progress);
  void setStatus(const std::string& status);

};

#endif 