#include "OpenSlideImage.h"
#include <openslide.h>
#include <sstream>

using namespace pathology;

OpenSlideImage::OpenSlideImage() : MultiResolutionImage(), _slide(NULL), _ignoreAlpha(true) {
}

OpenSlideImage::~OpenSlideImage() {
  boost::unique_lock<boost::shared_mutex> l(_openCloseMutex);
  cleanup();
  MultiResolutionImage::cleanup();
}

const bool OpenSlideImage::getIgnoreAlpha() const {
  return _ignoreAlpha;
}

void OpenSlideImage::setCacheSize(const unsigned long long cacheSize) {
  MultiResolutionImage::setCacheSize(cacheSize);
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

void OpenSlideImage::setIgnoreAlpha(const bool ignoreAlpha) {
  if (ignoreAlpha) {
    _samplesPerPixel = 3;
    _colorType = RGB;
  }
  else {
    _samplesPerPixel = 4;
    _colorType = ARGB;
  }
  _ignoreAlpha = ignoreAlpha;
}

bool OpenSlideImage::initialize(const std::string& imagePath) {
  boost::unique_lock<boost::shared_mutex> l(_openCloseMutex);
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
      _dataType = UChar;
      if (_ignoreAlpha) {
        _samplesPerPixel = 3;
        _colorType = RGB;
      }
      else {
        _samplesPerPixel = 4;
        _colorType = ARGB;
      }
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
      _isValid = true;
      createCache<unsigned int>();
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
std::string OpenSlideImage::getOpenSlideProperty(const std::string& propertyName) {
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

  boost::shared_lock<boost::shared_mutex> l(_openCloseMutex);
  unsigned int* temp = new unsigned int[width*height];
  std::fill(temp,temp+width*height,0);
  std::vector<unsigned long long> levelDims = this->getLevelDimensions(level);
  unsigned long long levelW = levelDims[0];
  unsigned long long levelH = levelDims[1];

  std::stringstream ssmH; 
  ssmH << "openslide.level[";
  ssmH << level;
  ssmH << "].tile-height";

  std::stringstream ssmW; 
  ssmW << "openslide.level[";
  ssmW << level;
  ssmW << "].tile-width";

  unsigned int tileW = openslide_get_property_value(_slide, ssmW.str().c_str()) ? atoi(openslide_get_property_value(_slide, ssmW.str().c_str())) : 512;
  unsigned int tileH = openslide_get_property_value(_slide, ssmW.str().c_str()) ? atoi(openslide_get_property_value(_slide, ssmH.str().c_str())) : 512;
  unsigned int levelDownsample = getLevelDownsample(level);

  long long levelStartX = startX/levelDownsample;
  long long levelStartY = startY/levelDownsample;
  long long startTileY = levelStartY-(levelStartY-((levelStartY/tileH)*tileH));
  long long startTileX = levelStartX-(levelStartX-((levelStartX/tileW)*tileW));
  long long finalX = levelStartX + width >= levelW ? levelW : levelStartX + width;
  long long finalY = levelStartY + height >= levelH ? levelH : levelStartY + height;

  for (long long iy= startTileY; iy<finalY;iy += tileH) {
    if (iy < 0) {
      continue;
    }
    for (long long ix= startTileX; ix<finalX; ix+= tileW)  {
      if (ix < 0) {
        continue;
      }
      std::stringstream k;
      k << ix*levelDownsample << "-" << iy*levelDownsample << "-" << level;
      unsigned int* tile = NULL;
      bool deleteTile = false;
      unsigned int cachedTileSize = 0;
      if (_cacheSize == 0) { // OpenSlide is thread-safe, so special case when we do not need caching;
        tile = new unsigned int[tileW*tileH];
        openslide_read_region(_slide, tile, ix*levelDownsample, iy*levelDownsample, level, tileW, tileH);
        deleteTile = true;
      }
      else {
        _cacheMutex.lock();
        std::static_pointer_cast<TileCache<unsigned int> >(_cache)->get(k.str(), tile, cachedTileSize);
        _cacheMutex.unlock();
        if (!tile) {
          tile = new unsigned int[tileW*tileH];
          openslide_read_region(_slide, tile, ix*levelDownsample, iy*levelDownsample, level, tileW, tileH);

          // If tile did not fit in the cache, delete it at the end
          _cacheMutex.lock();
          if (std::static_pointer_cast<TileCache<unsigned int> >(_cache)->set(k.str(), tile, tileW*tileH*sizeof(unsigned int))) {
            deleteTile = true;
          }
          _cacheMutex.unlock();
        }
      }
      if (!tile) {
        break;
      }
      long long ixx = (ix-levelStartX);
      long long iyy = (iy-levelStartY);
      long long lyh = levelStartY + height;
      long long lxw = levelStartX + width;
      long long ixw = ixx + tileW;
      long long rowLength = ixw > width ? tileW - (ixw - width) : tileW;
      long long tileDeltaX = 0;
      if (ixx < 0) {
        rowLength += ixx;
        tileDeltaX -= ixx;
        ixx = 0;        
      }
      for (unsigned int ty = 0; ty < tileH; ++ty) {
        if ((iyy + ty >= 0) && (ixx >= 0) && (iyy + ty < height) && lxw > 0){
          long long idx = (ty+iyy)*width + ixx;
          long long tids = (ty*tileW);
          std::copy(tile+tids+tileDeltaX, tile+tids+rowLength+tileDeltaX, temp + idx);
        } 
      }
      if (deleteTile) {
        delete[] tile;
      }
    }
  }
  if (_ignoreAlpha) {
    unsigned char* rgb = new unsigned char[width*height*3];
    unsigned char* bgra = (unsigned char*)temp;
    for (unsigned long long i = 0, j = 0; i < width*height*4; i+=4, j+=3) {
      rgb[j] = bgra[i + 2];
      rgb[j + 1] = bgra[i + 1];
      rgb[j + 2] = bgra[i];
    }
    delete[] temp;
    return rgb;
  }
  return temp;
}

void OpenSlideImage::cleanup() {
  if (_slide) {
    openslide_close(_slide);
    _slide = NULL;
  }
}