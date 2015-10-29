#ifndef _ArithmeticWholeSlideFilter
#define _ArithmeticWholeSlideFilter

#include "config/pathology_config.h"
#include <string>
#include <vector>
#include <map>

class MultiResolutionImage;
class ProgressMonitor;

class EXPORT_WHOLESLIDEFILTERS ArithmeticWholeSlideFilter {

private:
  MultiResolutionImage* _input;
  ProgressMonitor* _monitor;
  unsigned int _processedLevel;
  std::string _outPath;
  std::string _expression;

public:
  ArithmeticWholeSlideFilter();
  virtual ~ArithmeticWholeSlideFilter();

  void setInput(MultiResolutionImage* const input);
  void setProcessedLevel(const unsigned int processedLevel);
  unsigned int getProcessedLevel();
  void setProgressMonitor(ProgressMonitor* progressMonitor);
  ProgressMonitor* getProgressMonitor();
  bool process();
  void setOutput(const std::string& outPath);
  
  void setExpression(const std::string& expression);
  std::string getExpression() const;

};

#endif