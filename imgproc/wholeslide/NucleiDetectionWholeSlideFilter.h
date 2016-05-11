#ifndef _NUCLEIDETECTIONWHOLESLIDEFILTER
#define _NUCLEIDETECTIONWHOLESLIDEFILTER

#include "wholeslidefilters_export.h"
#include <string>
#include <vector>
#include <memory>

class MultiResolutionImage;
class ProgressMonitor;

class WHOLESLIDEFILTERS_EXPORT NucleiDetectionWholeSlideFilter {

private:
  std::weak_ptr<MultiResolutionImage> _input;
  ProgressMonitor* _monitor;
  unsigned int _processedLevel;
  std::string _outPath;
  float _threshold;
  float _alpha;
  float _beta;
  float _minRadius;
  float _maxRadius;
  float _stepRadius;
  std::vector<std::vector<float> > _centerPoints;

public:
  NucleiDetectionWholeSlideFilter();
  virtual ~NucleiDetectionWholeSlideFilter();

  float getAlpha();
  void setAlpha(const float& alpha);
  float getBeta();
  void setBeta(const float& beta);
  float getThreshold();
  void setThreshold(const float& threshold);

  void  setMaximumRadius(const float& maxRadius);
  float getMaximumRadius();
  void  setMinimumRadius(const float& minRadius);
  float getMinimumRadius();
  void  setRadiusStep(const float& stepRadius);
  float getRadiusStep();

  void setInput(const std::shared_ptr<MultiResolutionImage>& input);
  void setProcessedLevel(const unsigned int processedLevel);
  unsigned int getProcessedLevel();
  void setProgressMonitor(ProgressMonitor* progressMonitor);
  ProgressMonitor* getProgressMonitor();
  bool process();
  void setOutput(const std::string& outPath);
  std::vector<std::vector<float> > getCenterPoints();

};

#endif