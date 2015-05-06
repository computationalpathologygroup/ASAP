#ifndef CMDLINEPROGRESSMONITOR_H
#define CMDLINEPROGRESSMONITOR_H

#include "config/pathology_config.h"
#include "ProgressMonitor.h"
#include <boost/progress.hpp>

class EXPORT_CORE CmdLineProgressMonitor : public ProgressMonitor {

protected:
  boost::progress_display* disp;

public :

  CmdLineProgressMonitor();
  virtual ~CmdLineProgressMonitor();

  void setProgress(const unsigned int& progress);
  void setStatus(const std::string& status);

};

#endif 