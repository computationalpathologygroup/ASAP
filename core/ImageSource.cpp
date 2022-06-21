#include "ImageSource.h"

using namespace pathology;

ImageSource::ImageSource() :
  _spacing(),
  _samplesPerPixel(0),
  _colorType(ColorType::InvalidColorType),
  _dataType(DataType::InvalidDataType),
  _isValid(false)
{
}

ImageSource::~ImageSource() {
}

void ImageSource::swap(ImageSource& first, ImageSource& second) {
  std::swap(first._colorType, second._colorType);
  std::swap(first._dataType, second._dataType);
  std::swap(first._spacing, second._spacing);
  std::swap(first._samplesPerPixel, second._samplesPerPixel);
  std::swap(first._isValid, second._isValid);
}

const int ImageSource::getSamplesPerPixel() const {
  if (_isValid) {
    return _samplesPerPixel;
  } else {
    return -1;
  }
}

const ColorType ImageSource::getColorType() const {
  if (_isValid) {
    return _colorType;
  } else {
    return ColorType::InvalidColorType;
  }
}

const std::vector<double> ImageSource::getSpacing() const {
  return _spacing;
}

const DataType ImageSource::getDataType() const {
  if (_isValid) {
    return _dataType;
  } else {
    return DataType::InvalidDataType;
  }
 }