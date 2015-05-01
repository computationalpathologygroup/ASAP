#ifndef _ImageFilter
#define _ImageFilter

#include "core/ImageSource.h"
#include "core/Patch.h"
#include "FilterBase.h"

class ProgressMonitor;

template <class inType, class outType>
class ImageFilter : public ImageSource, public FilterBase {
  
  virtual bool checkInputImageRequirements(const Patch<inType>& input) const {return true;}
  virtual bool calculate(const Patch<inType>& input, Patch<outType>& output) = 0;

public :

  ImageFilter() : ImageSource(), FilterBase() {

  }

  virtual ~ImageFilter() 
  {
  };

  virtual std::string name() const {return std::string("ImageFilter");};

  bool filter(const Patch<inType>& input, Patch<outType>& output) {
    if (checkInputImageRequirements(input)) {
      start();
      bool result = calculate(input, output);
      finish();
      return result;
    } else {
      return false;
    }
  }

  virtual double getMinValue(int channel = -1) { return std::numeric_limits<double>::min(); };
  virtual double getMaxValue(int channel = -1) { return std::numeric_limits<double>::max(); };
  
  virtual const std::vector<unsigned long long> getDimensions() const {
    return std::vector<unsigned long long>(); 
  }

  virtual bool filter(unsigned int width, unsigned int height, unsigned int channels, pathology::ColorType ctype, inType *data, Patch<outType>& output) {
    std::vector<unsigned long long> dims;
    dims.push_back(width);
    dims.push_back(height);
    dims.push_back(channels);
    Patch<inType> input(dims, ctype, data, false);
    return filter(input, output);
  }

};

#endif