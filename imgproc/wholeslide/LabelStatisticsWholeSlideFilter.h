#ifndef _LabelStatisticsWholeSlideFilter
#define _LabelStatisticsWholeSlideFilter

#include "wholeslidefilters_export.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

class MultiResolutionImage;
class ProgressMonitor;

class WHOLESLIDEFILTERS_EXPORT LabelStatisticsWholeSlideFilter {

private:
  std::weak_ptr<MultiResolutionImage> _input;
  ProgressMonitor* _monitor;
  unsigned int _processedLevel;
  std::string _outPath;
  std::vector<std::vector<float> > _labelStats;

public:
  LabelStatisticsWholeSlideFilter();
  virtual ~LabelStatisticsWholeSlideFilter();

  void setInput(const std::shared_ptr<MultiResolutionImage>& input);
  void setProcessedLevel(const unsigned int processedLevel);
  unsigned int getProcessedLevel();
  void setProgressMonitor(ProgressMonitor* progressMonitor);
  ProgressMonitor* getProgressMonitor();
  bool process();
  void setOutput(const std::string& outPath);
  std::vector<std::vector<float> > getLabelStatistics();

};

#endif