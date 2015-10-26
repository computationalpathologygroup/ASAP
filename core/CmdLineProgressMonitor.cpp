#include "CmdLineProgressMonitor.h"

CmdLineProgressMonitor::CmdLineProgressMonitor() : ProgressMonitor(), disp(NULL)
{
}

CmdLineProgressMonitor::~CmdLineProgressMonitor() {
  if (disp) {
    delete disp;
    disp = NULL;
  }
}

void CmdLineProgressMonitor::setProgress(const unsigned int& progress) {
  ProgressMonitor::setProgress(progress);
  if (!disp) {
    disp = new boost::progress_display(100);
  } else if (progress == 0) {
    disp->restart(100);
  }
  (*disp) += ((100 * static_cast<float>(progress) / static_cast<float>(_maxProgress)) - disp->count());
}

void CmdLineProgressMonitor::setStatus(const std::string& status) {
  ProgressMonitor::setStatus(status);
  std::cout << status << std::endl;
}