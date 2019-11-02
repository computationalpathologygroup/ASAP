#include "OpenJP2Image.h"
#include "core/PathologyEnums.h"
#include <boost/thread.hpp>
#include "JPEG2000Codec.h"

using namespace pathology;

OpenJP2Image::OpenJP2Image() : MultiResolutionImage(), _stream(NULL) {
}

OpenJP2Image::~OpenJP2Image() {
  boost::unique_lock<boost::shared_mutex> l(*_openCloseMutex);
  cleanup();
  MultiResolutionImage::cleanup();
}

bool OpenJP2Image::initializeType(const std::string& imagePath) {
  boost::unique_lock<boost::shared_mutex> l(*_openCloseMutex);
  cleanup();
  _isValid = false;
  return _isValid;
}

double OpenJP2Image::getMinValue(int channel) {
  if (!_minValues.empty() && channel > 0 && channel < _minValues.size()) {
    return _minValues[channel];
  }
  else if (!_maxValues.empty()) {
    return *std::max_element(_minValues.begin(), _minValues.end());
  }
  else {
    return std::numeric_limits<double>::min();
  }
}

double OpenJP2Image::getMaxValue(int channel) {
  if (!_maxValues.empty() && channel > 0 && channel < _maxValues.size()) {
    return _maxValues[channel];
  }
  else if (!_maxValues.empty()) {
    return *std::max_element(_maxValues.begin(), _maxValues.end());
  }
  else {
    return std::numeric_limits<double>::max();
  }
}

void OpenJP2Image::cleanup() {  
}

void* OpenJP2Image::readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level) {
  if (getDataType()==UInt32) {
    unsigned int* temp = FillRequestedRegion<unsigned int>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else if (getDataType()==UInt16) {
    unsigned short* temp = FillRequestedRegion<unsigned short>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else if (getDataType()==Float) {
    float* temp = FillRequestedRegion<float>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else if (getDataType()==UChar) {
    unsigned char* temp = FillRequestedRegion<unsigned char>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else {
    return NULL;
  }
}

template <typename T> T* OpenJP2Image::FillRequestedRegion(const long long& startX, const long long& startY, const unsigned long long& width,
                                                           const unsigned long long& height, const unsigned int& level, unsigned int nrSamples)
{
  return NULL;
}