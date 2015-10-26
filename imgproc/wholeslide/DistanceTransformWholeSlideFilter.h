#ifndef _DistanceTransformWholeSlideFilter
#define _DistanceTransformWholeSlideFilter

#include "config/pathology_config.h"
#include <string>
#include <vector>

class MultiResolutionImage;
class ProgressMonitor;

class EXPORT_WHOLESLIDEFILTERS DistanceTransformWholeSlideFilter {

private:
  MultiResolutionImage* _input;
  ProgressMonitor* _monitor;
  unsigned int _processedLevel;
  std::string _outPath;

public:
  DistanceTransformWholeSlideFilter();
  virtual ~DistanceTransformWholeSlideFilter();

  void setInput(MultiResolutionImage* const input);
  void setProcessedLevel(const unsigned int processedLevel);
  unsigned int getProcessedLevel();
  void setProgressMonitor(ProgressMonitor* progressMonitor);
  ProgressMonitor* getProgressMonitor();
  bool process() const;
  void setOutput(const std::string& outPath);

};

#endif