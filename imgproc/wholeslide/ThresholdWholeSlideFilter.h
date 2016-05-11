#ifndef _ThresholdWholeSlideFilter
#define _ThresholdWholeSlideFilter

#include "wholeslidefilters_export.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

class MultiResolutionImage;
class ProgressMonitor;

class WHOLESLIDEFILTERS_EXPORT ThresholdWholeSlideFilter {

private:
  std::weak_ptr<MultiResolutionImage> _input;
  ProgressMonitor* _monitor;
  unsigned int _processedLevel;
  std::string _outPath;
  float _lowerThreshold;
  float _upperThreshold;
  int _component;
  bool _keepOrgValues;

public:
  ThresholdWholeSlideFilter();
  virtual ~ThresholdWholeSlideFilter();

  void setInput(const std::shared_ptr<MultiResolutionImage>& input);
  void setProcessedLevel(const unsigned int processedLevel);
  unsigned int getProcessedLevel();
  void setProgressMonitor(ProgressMonitor* progressMonitor);
  ProgressMonitor* getProgressMonitor();
  bool process();
  void setOutput(const std::string& outPath);
  
  void setLowerThreshold(const float& threshold);
  void setUpperThreshold(const float& threshold);

  float getLowerThreshold() const;
  float getUpperThreshold() const;

  void setComponent(const int& component);
  int getComponent() const;

};

#endif