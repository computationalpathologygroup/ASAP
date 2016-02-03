#ifndef _ThresholdWholeSlideFilter
#define _ThresholdWholeSlideFilter

#include "config/pathology_config.h"
#include <string>
#include <vector>
#include <map>

class MultiResolutionImage;
class ProgressMonitor;

class EXPORT_WHOLESLIDEFILTERS ThresholdWholeSlideFilter {

private:
  MultiResolutionImage* _input;
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

  void setInput(MultiResolutionImage* const input);
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