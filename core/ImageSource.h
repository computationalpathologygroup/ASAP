#ifndef _ImageSource
#define _ImageSource
#include <string>
#include <vector>
#include "core_export.h"
#include "core/PathologyEnums.h"

class CORE_EXPORT ImageSource {

public :
  ImageSource();
  virtual ~ImageSource();

  virtual bool valid() const {return _isValid;}

  //! Gets the dimensions of the base level of the pyramid
  virtual const std::vector<unsigned long long> getDimensions() const = 0;
  
  //! Returns the color type
  virtual const pathology::ColorType getColorType() const;

  //! Returns the data type
  virtual const pathology::DataType getDataType() const;

  //! Returns the number of samples per pixel
  virtual const int getSamplesPerPixel() const;

  //! Returns the pixel spacing meta-data (um per pixel)
  virtual const std::vector<double> getSpacing() const; 

  //! Gets the minimum value for a channel. If no channel is specified, default to overall minimum
  virtual double getMinValue(int channel = -1) = 0;

  //! Gets the maximum value for a channel. If no channel is specified, default to overall maximum
  virtual double getMaxValue(int channel = -1) = 0;

  void swap(ImageSource& first, ImageSource& second);

protected :

  // Properties of an image
  std::vector<double> _spacing;
  unsigned int _samplesPerPixel;
  pathology::ColorType _colorType;
  pathology::DataType _dataType;
  bool _isValid;

};

#endif