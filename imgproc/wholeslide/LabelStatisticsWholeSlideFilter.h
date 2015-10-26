#ifndef _LabelStatisticsWholeSlideFilter
#define _LabelStatisticsWholeSlideFilter

#include "config/pathology_config.h"
#include <string>
#include <vector>
#include <map>

class MultiResolutionImage;
class ProgressMonitor;

class EXPORT_WHOLESLIDEFILTERS LabelStatisticsWholeSlideFilter {

private:
  MultiResolutionImage* _input;
  ProgressMonitor* _monitor;
  unsigned int _processedLevel;
  std::string _outPath;
  std::vector<std::vector<float> > _labelStats;

public:
  LabelStatisticsWholeSlideFilter();
  virtual ~LabelStatisticsWholeSlideFilter();

  void setInput(MultiResolutionImage* const input);
  void setProcessedLevel(const unsigned int processedLevel);
  unsigned int getProcessedLevel();
  void setProgressMonitor(ProgressMonitor* progressMonitor);
  ProgressMonitor* getProgressMonitor();
  bool process();
  void setOutput(const std::string& outPath);
  std::vector<std::vector<float> > getLabelStatistics();

};

#endif