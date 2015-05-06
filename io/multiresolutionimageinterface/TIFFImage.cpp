#include "TIFFImage.h"
#include "tiffio.h"
#include "core/PathologyEnums.h"

using namespace pathology;

TIFFImage::TIFFImage() : MultiResolutionImage(), _tiff(NULL) {
}

TIFFImage::~TIFFImage() {
  boost::unique_lock<boost::shared_mutex> l(_openCloseMutex);
  cleanup();
  MultiResolutionImage::cleanup();
}

bool TIFFImage::initialize(const std::string& imagePath) {
  boost::unique_lock<boost::shared_mutex> l(_openCloseMutex);
  cleanup();
  
  //Disable warning popups
  _tiff = TIFFOpen(imagePath.c_str(),"rm");
  TIFFSetWarningHandler(NULL);
  TIFFSetErrorHandler(NULL);

  if (_tiff) {
  unsigned int cType = 0, dType = 0, planarconfig=0, bitsPerSample = 0;
  TIFFGetField(_tiff, TIFFTAG_PHOTOMETRIC, &cType);
  
  // TIFFTAG_SAMPLEFORMAT is not always present, when it is not the TIFF
  // standard indicates that data should be interpreted as unsigned int
  if (TIFFGetField(_tiff, TIFFTAG_SAMPLEFORMAT, &dType) == 0 && dType == 0) {
    dType = 1;
  }

  TIFFGetField(_tiff, TIFFTAG_SAMPLESPERPIXEL, &_samplesPerPixel);
  // Going to use TIFFReadRGBA for cType == PHOTOMETRIC_YCBCR, so four samples
  if (cType == PHOTOMETRIC_YCBCR) {
    _samplesPerPixel = 4;
  }

  TIFFGetField(_tiff, TIFFTAG_BITSPERSAMPLE, & bitsPerSample);
  TIFFGetField(_tiff, TIFFTAG_PLANARCONFIG, &planarconfig);

  if (planarconfig!=PLANARCONFIG_CONTIG) {
    cleanup();
  }
  if (dType != SAMPLEFORMAT_IEEEFP && dType != SAMPLEFORMAT_UINT) {
    cleanup();
  }
  if (cType != PHOTOMETRIC_MINISBLACK && cType != PHOTOMETRIC_RGB && cType != PHOTOMETRIC_YCBCR) {
    cleanup();
  }
  if (cType == PHOTOMETRIC_RGB && _samplesPerPixel != 3 && _samplesPerPixel != 4) {
    cleanup();
  }
  if (dType == SAMPLEFORMAT_IEEEFP && bitsPerSample != 32) {
    cleanup();
  }
  if (dType == SAMPLEFORMAT_UINT && bitsPerSample != 32 && bitsPerSample != 16 &&bitsPerSample != 8) {
    cleanup();
  }
  if (!_tiff) {
    return false;
  }

  // Check the multi-resolution pyramid is stored nicely
  TIFFSetDirectory(_tiff,0);
  unsigned int width = 0;
  TIFFGetField(_tiff, TIFFTAG_IMAGEWIDTH, &width);
  unsigned int x = 0, y = 0;
  unsigned int tileW = 0, tileH = 0;
  unsigned int dir = 0;
  for (int level = 0; level < TIFFNumberOfDirectories(_tiff); ++level) {
    TIFFSetDirectory(_tiff, level);
    TIFFGetField(_tiff, TIFFTAG_IMAGEWIDTH, &x);
    TIFFGetField(_tiff, TIFFTAG_IMAGELENGTH, &y);
    TIFFGetField(_tiff, TIFFTAG_TILEWIDTH, &tileW);
    TIFFGetField(_tiff, TIFFTAG_TILELENGTH, &tileH);
    std::vector<unsigned long long> tmp;
    std::vector<unsigned int> levelTileSize;
    tmp.push_back(x);
    tmp.push_back(y);
    levelTileSize.push_back(tileW);
    levelTileSize.push_back(tileH);
    _levelDimensions.push_back(tmp);   
    _tileSizesPerLevel.push_back(levelTileSize);
    if (level > 0) {
      if (width > x) {
        width = x;        
      } else {
        cleanup();
        return false;
      }
    }
  }

  _numberOfLevels = TIFFNumberOfDirectories(_tiff);

  if (dType == SAMPLEFORMAT_IEEEFP) {
    _dataType = Float;
  }
  else if (dType == SAMPLEFORMAT_UINT) {
    if (bitsPerSample==8) {
      _dataType =  UChar;
    } else if (bitsPerSample==16) {
      _dataType =  UInt16;
    } else if (bitsPerSample==32) {
      _dataType =  UInt32;
    } else {
      _dataType =  InvalidDataType;
    }
  }

  if (cType == PHOTOMETRIC_MINISBLACK) {
    if (_samplesPerPixel > 1) {
      _colorType = Indexed;
    } else {
      _colorType = Monochrome;
    }
  }
  else if (cType == PHOTOMETRIC_RGB) {
    if (_samplesPerPixel==3) {
      _colorType =  RGB;
    } else if (_samplesPerPixel==4) {
      _colorType = ARGB;
    } else {
      _colorType = InvalidColorType;
    }
  }  else if (cType == PHOTOMETRIC_YCBCR) {
    if (_samplesPerPixel==4) {
      _colorType =  ARGB;
    } else {
      _colorType = InvalidColorType;
    }
  }

  TIFFSetDirectory(_tiff, 0);
  float spacingX;
  float spacingY;
  if (!TIFFGetField(_tiff, TIFFTAG_XRESOLUTION, &spacingX)) {
    spacingX = 1.;
  }
  if (!TIFFGetField(_tiff, TIFFTAG_YRESOLUTION, &spacingY)) {
    spacingY = 1.; 
  }
  _spacing[0] = 1. / (spacingX / (10000.));
  _spacing[1] = 1. / (spacingY / (10000.));

  TIFFSetField(_tiff, TIFFTAG_PERSAMPLE, PERSAMPLE_MULTI);
  double* min_values;  
  if (TIFFGetField(_tiff, TIFFTAG_SMINSAMPLEVALUE, &min_values)) {
    for (int i = 0; i < _samplesPerPixel; ++i) {
      _minValues.push_back(min_values[i]);
    }
  }
  else {
    for (int i = 0; i < _samplesPerPixel; ++i) {
      _minValues.push_back(0.);
    }
  }
  double* max_values;
  if (TIFFGetField(_tiff, TIFFTAG_SMAXSAMPLEVALUE, &max_values)) {
    for (int i = 0; i < _samplesPerPixel; ++i) {
      _maxValues.push_back(max_values[i]);
    }
  }
  else {
    for (int i = 0; i < _samplesPerPixel; ++i) {
      _maxValues.push_back(255.);
    }
  }
  TIFFSetField(_tiff, TIFFTAG_PERSAMPLE, PERSAMPLE_MERGED);

  _fileType = "tif";
  _isValid = true;
} else {
  cleanup();
  return false;
}

if (_dataType == UInt32) {
  createCache<unsigned int>();
} 
else if (_dataType == UInt16){
  createCache<unsigned short>();
} 
else if (_dataType == Float){ 
  createCache<float>();
} 
else if (_dataType == UChar) {
  createCache<unsigned char>();
}
return _isValid;
}

double TIFFImage::getMinValue(int channel) {
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

double TIFFImage::getMaxValue(int channel) {
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

void TIFFImage::cleanup() {
  _tileSizesPerLevel.clear();
  if (_tiff) {
    TIFFClose(_tiff);
    _tiff = NULL;
  }
}

void* TIFFImage::readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level) {
  if (getDataType()==UInt32) {
    unsigned int* temp = FillRequestedRegionFromTIFF<unsigned int>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else if (getDataType()==UInt16) {
    unsigned short* temp = FillRequestedRegionFromTIFF<unsigned short>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else if (getDataType()==Float) {
    float* temp = FillRequestedRegionFromTIFF<float>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else if (getDataType()==UChar) {
    unsigned char* temp = FillRequestedRegionFromTIFF<unsigned char>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else {
    return NULL;
  }
}

template <typename T> T* TIFFImage::FillRequestedRegionFromTIFF(const long long& startX, const long long& startY, const unsigned long long& width, 
                                                                                 const unsigned long long& height, const unsigned int& level, unsigned int nrSamples)
{
  boost::shared_lock<boost::shared_mutex> l(_openCloseMutex);
  T* temp = new T[width*height*nrSamples];
  std::fill(temp,temp+width*height*nrSamples,0);
  unsigned int tileW=_tileSizesPerLevel[level][0], tileH=_tileSizesPerLevel[level][1], levelH=_levelDimensions[level][1], levelW=_levelDimensions[level][0];

  long long levelStartX = startX / getLevelDownsample(level);
  long long levelStartY = startY / getLevelDownsample(level);
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
      k << ix*getLevelDownsample(level) << "-" << iy*getLevelDownsample(level) << "-" << level;
      bool deleteTile = false;
      unsigned int cachedTileSize = 0;
      T* tile = NULL;
      _cacheMutex.lock();
      boost::static_pointer_cast<TileCache<T> >(_cache)->get(k.str(), tile, cachedTileSize);
      _cacheMutex.unlock();
      if (!tile) {
        tile = new T[tileW*tileH*getSamplesPerPixel()];
        std::fill(tile, tile + tileW*tileH*getSamplesPerPixel(), static_cast<T>(0.0));
        _cacheMutex.lock();
        TIFFSetDirectory(_tiff, level);      
        unsigned int ycbcr = 0;
        TIFFGetField(_tiff, TIFFTAG_PHOTOMETRIC, &ycbcr);
        if (ycbcr == PHOTOMETRIC_YCBCR) {
          TIFFReadRGBATile(_tiff, ix, iy, (uint32*)tile);
        }
        else {
          TIFFReadTile(_tiff, tile, ix, iy, 0, 0);
        }
        if (boost::static_pointer_cast<TileCache<T> >(_cache)->set(k.str(), tile, tileW*tileH*getSamplesPerPixel()*sizeof(T))) {
          deleteTile = true;
        }
        _cacheMutex.unlock();
      }

      long long ixx = (ix-levelStartX);
      long long iyy = (iy-levelStartY);
      long long lyh = levelStartY + height;
      long long lxw = levelStartX + width;
      long long ixw = ixx + tileW;
      long long rowLength = ixw > width ? (tileW - (ixw - width))*nrSamples : tileW*nrSamples;
      long long tileDeltaX = 0;
      if (ixx < 0) {
        rowLength += ixx*nrSamples;
        tileDeltaX -= ixx*nrSamples;
        ixx = 0;        
      }
      for (unsigned int ty = 0; ty < tileH; ++ty) {
        if ((iyy + ty >= 0) && (ixx >= 0) && (iyy + ty < height) && lxw > 0){
          long long idx = (ty+iyy)*width*nrSamples + ixx*nrSamples;
          long long tids = (ty*tileW)*nrSamples;
          std::copy(tile+tids+tileDeltaX, tile+tids+rowLength+tileDeltaX, temp + idx);
        } 
      }
      if (deleteTile) {
        delete[] tile;
      }
    }
  }
  return temp;
}