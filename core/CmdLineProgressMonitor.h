#ifndef CMDLINEPROGRESSMONITOR_H
#define CMDLINEPROGRESSMONITOR_H

#include "core_export.h"
#include "ProgressMonitor.h"

namespace boost {
  class progress_display;
}

class CORE_EXPORT CmdLineProgressMonitor : public ProgressMonitor {

protected:
  boost::progress_display* disp;

public :

  CmdLineProgressMonitor();
  virtual ~CmdLineProgressMonitor();

  void setProgress(const unsigned int& progress);
  void setStatus(const std::string& status);

};

#endif 