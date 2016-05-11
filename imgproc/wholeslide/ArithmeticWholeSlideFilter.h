#ifndef _ArithmeticWholeSlideFilter
#define _ArithmeticWholeSlideFilter

#include "wholeslidefilters_export.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

class MultiResolutionImage;
class ProgressMonitor;

class WHOLESLIDEFILTERS_EXPORT ArithmeticWholeSlideFilter {

private:
  std::weak_ptr<MultiResolutionImage> _input;
  ProgressMonitor* _monitor;
  unsigned int _processedLevel;
  std::string _outPath;
  std::string _expression;

public:
  ArithmeticWholeSlideFilter();
  virtual ~ArithmeticWholeSlideFilter();

  void setInput(const std::shared_ptr<MultiResolutionImage>& input);
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