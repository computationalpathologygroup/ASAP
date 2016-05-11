#ifndef _DistanceTransformWholeSlideFilter
#define _DistanceTransformWholeSlideFilter

#include "wholeslidefilters_export.h"
#include <string>
#include <vector>
#include <memory>

class MultiResolutionImage;
class ProgressMonitor;

class WHOLESLIDEFILTERS_EXPORT DistanceTransformWholeSlideFilter {

private:
  std::weak_ptr<MultiResolutionImage> _input;
  ProgressMonitor* _monitor;
  unsigned int _processedLevel;
  std::string _outPath;

public:
  DistanceTransformWholeSlideFilter();
  virtual ~DistanceTransformWholeSlideFilter();

  void setInput(const std::shared_ptr<MultiResolutionImage>& input);
  void setProcessedLevel(const unsigned int processedLevel);
  unsigned int getProcessedLevel();
  void setProgressMonitor(ProgressMonitor* progressMonitor);
  ProgressMonitor* getProgressMonitor();
  bool process() const;
  void setOutput(const std::string& outPath);

};

#endif