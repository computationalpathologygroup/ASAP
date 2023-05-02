#include "OpenSlideImage.h"
#include <shared_mutex>
#include "openslide.h" 
#include <sstream>

using namespace pathology;

OpenSlideImage::OpenSlideImage() : MultiResolutionImage(), _slide(NULL), _bg_r(255), _bg_g(255), _bg_b(255) {
}

OpenSlideImage::~OpenSlideImage() {
  std::unique_lock<std::shared_mutex> l(*_openCloseMutex);
  cleanup();
  MultiResolutionImage::cleanup();
}

// We are using OpenSlides caching system instead of our own.
void OpenSlideImage::setCacheSize(const unsigned long long cacheSize) {
#ifdef CUSTOM_OPENSLIDE
  if (_slide) {
    openslide_set_cache_size(_slide, cacheSize);
  }
#endif
}

std::string OpenSlideImage::getOpenSlideErrorState() {
  if (_errorState.empty()) {
    return "No file opened.";
  }
  return _errorState;
}

bool OpenSlideImage::initializeType(const std::string& imagePath) {
  std::unique_lock<std::shared_mutex> l(*_openCloseMutex);
  cleanup();

  if (openslide_detect_vendor(imagePath.c_str())) {
    _slide = openslide_open(imagePath.c_str());
    if (const char* error = openslide_get_error(_slide)) {
      _errorState = error;
    }
    else {
      _errorState = "";
    }
    if (_errorState.empty()) {
      _numberOfLevels = openslide_get_level_count(_slide);
      _dataType = DataType::UChar;
      _samplesPerPixel = 3;
      _colorType = ColorType::RGB;
      for (int i = 0; i < _numberOfLevels; ++i) {
        int64_t x, y;
        openslide_get_level_dimensions(_slide, i, &x, &y);
        std::vector<unsigned long long> tmp;
        tmp.push_back(x);
        tmp.push_back(y);
        _levelDimensions.push_back(tmp);
      }
      std::stringstream ssm;
      if (openslide_get_property_value(_slide, OPENSLIDE_PROPERTY_NAME_MPP_X)) {
        ssm << openslide_get_property_value(_slide, OPENSLIDE_PROPERTY_NAME_MPP_X);
        float tmp;
        ssm >> tmp;
        _spacing.push_back(tmp);
        ssm.clear();
      }
      if (openslide_get_property_value(_slide, OPENSLIDE_PROPERTY_NAME_MPP_Y)) {
        ssm << openslide_get_property_value(_slide, OPENSLIDE_PROPERTY_NAME_MPP_Y);
        float tmp;
        ssm >> tmp;
        _spacing.push_back(tmp);
        ssm.clear();
      }
      _fileType = openslide_get_property_value(_slide, OPENSLIDE_PROPERTY_NAME_VENDOR);
      
      // Get background color if present
      const char* bg_color_hex = openslide_get_property_value(_slide, "openslide.background-color");
      if (bg_color_hex) {
        unsigned int bg_color = std::stoi(bg_color_hex, 0, 16);
        _bg_r = ((bg_color >> 16) & 0xff);
        _bg_g = ((bg_color >> 8) & 0xff);
        _bg_b = (bg_color & 0xff);
      }
      _isValid = true;
    }
    else {
      _isValid = false;
    }
  } 
  else {
    _isValid = false;
  }
  return _isValid;
}
std::string OpenSlideImage::getProperty(const std::string& propertyName) {
  std::string propertyValue;
  if (_slide) {
    if (openslide_get_property_value(_slide, propertyName.c_str())) {
      propertyValue = openslide_get_property_value(_slide, propertyName.c_str());
    }
  }
  return propertyValue;
}

void* OpenSlideImage::readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level) {
  
  if (!_isValid) {
    return NULL;
  }

  std::shared_lock<std::shared_mutex> l(*_openCloseMutex);
  unsigned int* temp = new unsigned int[width*height];
  openslide_read_region(_slide, temp, startX, startY, level, width, height);

  if (openslide_get_error(_slide)) {
    delete[] temp;
    return NULL;
  }

  unsigned char* rgb = new unsigned char[width*height*3];
  unsigned char* bgra = (unsigned char*)temp;
  for (unsigned long long i = 0, j = 0; i < width*height*4; i+=4, j+=3) {
    if (bgra[i + 3] == 255) {
      rgb[j] = bgra[i + 2];
      rgb[j + 1] = bgra[i + 1];
      rgb[j + 2] = bgra[i];
    }
    else if (bgra[i + 3] == 0) {
      rgb[j] = _bg_r;
      rgb[j + 1] = _bg_g;
      rgb[j + 2] = _bg_b;
    }
    else {
      rgb[j] = (255. * bgra[i + 2]) / bgra[i + 3];
      rgb[j + 1] = (255. * bgra[i + 1]) / bgra[i + 3];
      rgb[j + 2] = (255. * bgra[i]) / bgra[i + 3];
    }
  }
  delete[] temp;
  return rgb;
}

void OpenSlideImage::cleanup() {
  if (_slide) {
    openslide_close(_slide);
    _slide = NULL;
  }
}