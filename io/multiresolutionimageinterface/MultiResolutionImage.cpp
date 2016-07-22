#include "MultiResolutionImage.h"
#include "boost/thread.hpp"

using namespace pathology;

// Subsequent specialization to not re-copy data when datatypes are the same
template <> void MultiResolutionImage::getRawRegion(const long long& startX, const long long& startY, const unsigned long long& width, 
  const unsigned long long& height, const unsigned int& level, float*& data) {
    if (level >= getNumberOfLevels()) {
      return;
    }
    unsigned int nrSamples = getSamplesPerPixel();
    if (this->getDataType()==pathology::Float) {
      delete[] data;
      data = (float*)readDataFromImage(startX, startY, width, height, level);
    }
    else if (this->getDataType()==pathology::UChar) {
      unsigned char * temp = (unsigned char*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
    else if (this->getDataType()==pathology::UInt16) {
      unsigned short * temp = (unsigned short*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
    else if (this->getDataType()==pathology::UInt32) {
      unsigned int * temp = (unsigned int*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
}

template <> void MultiResolutionImage::getRawRegion(const long long& startX, const long long& startY, const unsigned long long& width, 
  const unsigned long long& height, const unsigned int& level, unsigned char*& data) {
    if (level >= getNumberOfLevels()) {
      return;
    }
    unsigned int nrSamples = getSamplesPerPixel();
    if (this->getDataType()==pathology::Float) {
      float * temp = (float*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
    else if (this->getDataType()==pathology::UChar) {
      delete[] data;
      data = (unsigned char*)readDataFromImage(startX, startY, width, height, level);
    }
    else if (this->getDataType()==pathology::UInt16) {
      unsigned short * temp = (unsigned short*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
    else if (this->getDataType()==pathology::UInt32) {
      unsigned int * temp = (unsigned int*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
}

template <> void MultiResolutionImage::getRawRegion(const long long& startX, const long long& startY, const unsigned long long& width, 
  const unsigned long long& height, const unsigned int& level, unsigned short*& data) {
    if (level >= getNumberOfLevels()) {
      return;
    }
    unsigned int nrSamples = getSamplesPerPixel();
    if (this->getDataType()==pathology::Float) {
      float* temp = (float*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
    else if (this->getDataType()==pathology::UChar) {
      unsigned char * temp = (unsigned char*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
    else if (this->getDataType()==pathology::UInt16) {
      delete[] data;
      data = (unsigned short*)readDataFromImage(startX, startY, width, height, level);
    }
    else if (this->getDataType()==pathology::UInt32) {
      unsigned int* temp = (unsigned int*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
}

template <> void MultiResolutionImage::getRawRegion(const long long& startX, const long long& startY, const unsigned long long& width, 
  const unsigned long long& height, const unsigned int& level, unsigned int*& data) {
    if (level >= getNumberOfLevels()) {
      return;
    }
    unsigned int nrSamples = getSamplesPerPixel();
    if (this->getDataType()==pathology::Float) {
      float * temp = (float*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
    else if (this->getDataType()==pathology::UChar) {
      unsigned char * temp = (unsigned char*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
    else if (this->getDataType()==pathology::UInt16) {
      unsigned short * temp = (unsigned short*)readDataFromImage(startX, startY, width, height, level);
      std::copy(temp, temp + width*height*nrSamples, data);
      delete[] temp;
    }
    else if (this->getDataType()==pathology::UInt32) {
      delete[] data;
      data = (unsigned int*)readDataFromImage(startX, startY, width, height, level);
    }
}

MultiResolutionImage::MultiResolutionImage() :
  ImageSource(),
  _cacheSize(0),
  _cache(),
  _levelDimensions(),
  _numberOfLevels(0),
  _filePath(),
  _fileType()
{
  _cacheMutex.reset(new boost::mutex());
  _openCloseMutex.reset(new boost::shared_mutex());
}

const int MultiResolutionImage::getNumberOfLevels() const {
  if (_isValid) {
  return _numberOfLevels;
  }
  else {
    return -1;
  }
}

const std::vector<unsigned long long> MultiResolutionImage::getDimensions() const {
  std::vector<unsigned long long> dims;
  if (_isValid) {
    return _levelDimensions[0];
  }
  return dims;
}

bool MultiResolutionImage::initialize(const std::string& imagePath) {
  _filePath = imagePath;
  return initializeType(imagePath);
}

const std::vector<unsigned long long> MultiResolutionImage::getLevelDimensions(const unsigned int& level) const {
  std::vector<unsigned long long> dims;
  if (_isValid && (level < getNumberOfLevels())) {
    return _levelDimensions[level];
  }
  return dims;
}

const double MultiResolutionImage::getLevelDownsample(const unsigned int& level) const {
  if (_isValid && (level < getNumberOfLevels())) {
    return static_cast<float>(_levelDimensions[0][0])/_levelDimensions[level][0];
  }
  else {
    return -1.0;
  }
}

const int MultiResolutionImage::getBestLevelForDownSample(const double& downsample) const {
  if (_isValid) {
    float previousDownsample = 1.0;
    if (downsample < 1.0) {
      return 0;
    }
    for (int i = 1; i < _levelDimensions.size(); ++i) {
      double currentDownSample = (double)_levelDimensions[0][0]/(double)_levelDimensions[i][0];
      double previousDownSample = (double)_levelDimensions[0][0] / (double)_levelDimensions[i-1][0];
      if (downsample<currentDownSample) {
        
        if (abs(currentDownSample - downsample) > abs(previousDownSample - downsample)) {
          return i - 1;
        }
        else {
          return i;
        }
        
        return i - 1;
      }
    }
    return getNumberOfLevels()-1;
  }
  else {
    return -1;
  }
}

MultiResolutionImage::~MultiResolutionImage() {
  boost::unique_lock<boost::shared_mutex> l(*_openCloseMutex);
  cleanup();
}

void MultiResolutionImage::cleanup() {
  _levelDimensions.clear();
  _spacing.clear();
  _samplesPerPixel = 0;
  _numberOfLevels = 0;
  _colorType = InvalidColorType;
  _dataType = InvalidDataType;
  _isValid = false;
  _fileType = "";
  _filePath = "";
}

const unsigned long long MultiResolutionImage::getCacheSize() {
  unsigned long long cacheSize = 0;
  _cacheMutex->lock();
  if (_cache && _isValid) {
    if (_dataType == UInt32) {
      cacheSize = (std::static_pointer_cast<TileCache<unsigned int> >(_cache))->maxCacheSize();
    }
    else if (_dataType == UInt16) {
      cacheSize = (std::static_pointer_cast<TileCache<unsigned short> >(_cache))->maxCacheSize();
    }
    else if (_dataType == UChar) {
      cacheSize = (std::static_pointer_cast<TileCache<unsigned char> >(_cache))->maxCacheSize();
    }
    else if (_dataType == Float) {
      cacheSize = (std::static_pointer_cast<TileCache<float> >(_cache))->maxCacheSize();
    }
  _cacheMutex->unlock();
  }
  return cacheSize;
}

void MultiResolutionImage::setCacheSize(const unsigned long long cacheSize) {
  _cacheMutex->lock();
  if (_cache && _isValid) {
    if (_dataType == UInt32) {
      (std::static_pointer_cast<TileCache<unsigned int> >(_cache))->setMaxCacheSize(cacheSize);
    }
    else if (_dataType == UInt16) {
      (std::static_pointer_cast<TileCache<unsigned short> >(_cache))->setMaxCacheSize(cacheSize);
    }
    else if (_dataType == UChar) {
      (std::static_pointer_cast<TileCache<unsigned char> >(_cache))->setMaxCacheSize(cacheSize);
    }
    else if (_dataType == Float) {
      (std::static_pointer_cast<TileCache<float> >(_cache))->setMaxCacheSize(cacheSize);
    }
  _cacheMutex->unlock();
  }
}