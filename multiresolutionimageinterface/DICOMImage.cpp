#include "DICOMImage.h"
#include <shared_mutex>
#include <sstream>

using namespace pathology;

DICOMImage::DICOMImage() : MultiResolutionImage() {
}

DICOMImage::~DICOMImage() {
  std::unique_lock<std::shared_mutex> l(*_openCloseMutex);
  cleanup();
  MultiResolutionImage::cleanup();
}

// We are using OpenSlides caching system instead of our own.
void DICOMImage::setCacheSize(const unsigned long long cacheSize) {
#ifdef CUSTOM_OPENSLIDE
  if (_slide) {
    openslide_set_cache_size(_slide, cacheSize);
  }
#endif
}

bool DICOMImage::initializeType(const std::string& imagePath) {
  std::unique_lock<std::shared_mutex> l(*_openCloseMutex);
  cleanup();
  return _isValid;
}
std::string DICOMImage::getProperty(const std::string& propertyName) {
  std::string propertyValue;
  return propertyValue;
}

void* DICOMImage::readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level) {
  return nullptr;
}

void DICOMImage::cleanup() {
}