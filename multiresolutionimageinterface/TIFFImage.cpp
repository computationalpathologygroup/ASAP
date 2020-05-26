#include "TIFFImage.h"
#ifdef _WIN32
#include <Windows.h>
#include <Stringapiset.h>
#endif
#include "tiffio.h"
#include "JPEG2000Codec.h"
#include "core/PathologyEnums.h"
#include <boost/thread.hpp>

using namespace pathology;

TIFFImage::TIFFImage() : MultiResolutionImage(), _tiff(NULL), _jp2000(NULL) {
}

TIFFImage::~TIFFImage() {
  boost::unique_lock<boost::shared_mutex> l(*_openCloseMutex);
  cleanup();
  MultiResolutionImage::cleanup();
}

bool TIFFImage::initializeType(const std::string& imagePath) {
  boost::unique_lock<boost::shared_mutex> l(*_openCloseMutex);
  cleanup();

#ifdef _WIN32
  int wchars_num = MultiByteToWideChar(CP_UTF8, 0, imagePath.c_str(), -1, NULL, 0);
  wchar_t* w_imagePath = new wchar_t[wchars_num];
  MultiByteToWideChar(CP_UTF8, 0, imagePath.c_str(), -1, w_imagePath, wchars_num);
  _tiff = TIFFOpenW(w_imagePath, "rm");
  delete[] w_imagePath;
#else
  _tiff = TIFFOpen(imagePath.c_str(), "rm");
#endif

  if (_tiff) {
    const char* img_desc = NULL;

    if (TIFFGetFieldDefaulted(_tiff, TIFFTAG_IMAGEDESCRIPTION, &img_desc)) {
      std::string img_desc_string(img_desc);
      if (img_desc_string.find("DICOM") != std::string::npos || img_desc_string.find("xml") != std::string::npos || img_desc_string.find("XML") != std::string::npos) {
        // Not one of our TIFFs
        cleanup();
        return false;
      }
    }

    if (TIFFIsTiled(_tiff) == 0) {
      cleanup();
      return false;
    }
    unsigned int codec = 0;
    TIFFGetField(_tiff, TIFFTAG_COMPRESSION, &codec);
    if (codec != 33005 && codec != COMPRESSION_DEFLATE && codec != COMPRESSION_JPEG && codec != COMPRESSION_LZW && codec != COMPRESSION_NONE) {
      cleanup();
      return false;
    }

    unsigned int cType = 0, dType = 0, planarconfig = 0, bitsPerSample = 0;
    TIFFGetField(_tiff, TIFFTAG_PHOTOMETRIC, &cType);

    // TIFFTAG_SAMPLEFORMAT is not always present, when it is not the TIFF
    // standard indicates that data should be interpreted as unsigned int
    if (TIFFGetField(_tiff, TIFFTAG_SAMPLEFORMAT, &dType) == 0 && dType == 0) {
      dType = 1;
    }

    TIFFGetField(_tiff, TIFFTAG_SAMPLESPERPIXEL, &_samplesPerPixel);
    TIFFGetField(_tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(_tiff, TIFFTAG_PLANARCONFIG, &planarconfig);

    if (planarconfig != PLANARCONFIG_CONTIG) {
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
    if (dType == SAMPLEFORMAT_UINT && bitsPerSample != 32 && bitsPerSample != 16 && bitsPerSample != 8) {
      cleanup();
    }
    if (!_tiff) {
      return false;
    }

    _numberOfLevels = TIFFNumberOfDirectories(_tiff);

    if (_numberOfLevels < 1) {
      cleanup();
      return false;
    }

    // Check if the multi-resolution pyramid is stored nicely
    TIFFSetDirectory(_tiff, 0);
    unsigned int width = 0;
    TIFFGetField(_tiff, TIFFTAG_IMAGEWIDTH, &width);
    unsigned int x = 0, y = 0;
    unsigned int tileW = 0, tileH = 0;
    unsigned int dir = 0;
    for (int level = 0; level < TIFFNumberOfDirectories(_tiff); ++level) {
      TIFFSetDirectory(_tiff, level);
      if (TIFFIsTiled(_tiff) == 1) {
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
          }
          else {
            cleanup();
            return false;
          }
        }
      }
      else {
        _numberOfLevels -= 1;
      }
    }

    TIFFSetDirectory(_tiff, 0);
    if (dType == SAMPLEFORMAT_IEEEFP) {
      _dataType = Float;
    }
    else if (dType == SAMPLEFORMAT_UINT) {
      if (bitsPerSample == 8) {
        _dataType = UChar;
      }
      else if (bitsPerSample == 16) {
        _dataType = UInt16;
      }
      else if (bitsPerSample == 32) {
        _dataType = UInt32;
      }
      else {
        _dataType = InvalidDataType;
      }
    }

    if (cType == PHOTOMETRIC_MINISBLACK) {
      if (_samplesPerPixel > 1) {
        _colorType = Indexed;
      }
      else {
        _colorType = Monochrome;
      }
    }
    else if (cType == PHOTOMETRIC_RGB) {
      if (_samplesPerPixel == 3) {
        _colorType = RGB;
      }
      else if (_samplesPerPixel == 4) {
        _colorType = RGBA;
      }
      else {
        _colorType = InvalidColorType;
      }
    }
    else if (cType == PHOTOMETRIC_YCBCR && codec == COMPRESSION_JPEG) {
      if (_samplesPerPixel == 3) {
        _colorType = RGB;
      }
      else if (_samplesPerPixel == 4) {
        _colorType = RGBA;
      }
      else {
        _colorType = InvalidColorType;
      }
    }

    float spacingX;
    float spacingY;
    if (TIFFGetField(_tiff, TIFFTAG_XRESOLUTION, &spacingX) == 1) {
      _spacing.push_back(1. / (spacingX / (10000.)));
    }
    if (TIFFGetField(_tiff, TIFFTAG_YRESOLUTION, &spacingY) == 1) {
      _spacing.push_back(1. / (spacingY / (10000.)));
    }

    TIFFSetField(_tiff, TIFFTAG_PERSAMPLE, PERSAMPLE_MULTI);
    double* min_values;
    if (TIFFGetField(_tiff, TIFFTAG_SMINSAMPLEVALUE, &min_values)) {
      for (unsigned int i = 0; i < _samplesPerPixel; ++i) {
        _minValues.push_back(min_values[i]);
      }
    }
    else {
      for (unsigned int i = 0; i < _samplesPerPixel; ++i) {
        _minValues.push_back(0.);
      }
    }
    double* max_values;
    if (TIFFGetField(_tiff, TIFFTAG_SMAXSAMPLEVALUE, &max_values)) {
      for (unsigned int i = 0; i < _samplesPerPixel; ++i) {
        _maxValues.push_back(max_values[i]);
      }
    }
    else {
      for (unsigned int i = 0; i < _samplesPerPixel; ++i) {
        _maxValues.push_back(255.);
      }
    }
    TIFFSetField(_tiff, TIFFTAG_PERSAMPLE, PERSAMPLE_MERGED);

    _fileType = "tif";
    _isValid = true;
  }
  else {
    cleanup();
    return false;
  }

  if (_dataType == UInt32) {
    createCache<unsigned int>();
  }
  else if (_dataType == UInt16) {
    createCache<unsigned short>();
  }
  else if (_dataType == Float) {
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
  if (_jp2000) {
    delete _jp2000;
    _jp2000 = NULL;
  }
}

void* TIFFImage::readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width,
  const unsigned long long& height, const unsigned int& level) {
  if (getDataType() == UInt32) {
    unsigned int* temp = FillRequestedRegionFromTIFF<unsigned int>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else if (getDataType() == UInt16) {
    unsigned short* temp = FillRequestedRegionFromTIFF<unsigned short>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else if (getDataType() == Float) {
    float* temp = FillRequestedRegionFromTIFF<float>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else if (getDataType() == UChar) {
    unsigned char* temp = FillRequestedRegionFromTIFF<unsigned char>(startX, startY, width, height, level, _samplesPerPixel);
    return (void*)temp;
  }
  else {
    return NULL;
  }
}

long long TIFFImage::getEncodedTileSize(const long long& startX, const long long& startY, const unsigned int& level) {
  if (_tiff && level < this->_numberOfLevels) {
    long long levelStartX = std::floor(startX / getLevelDownsample(level) + 0.5);
    long long levelStartY = std::floor(startY / getLevelDownsample(level) + 0.5);
    TIFFSetDirectory(_tiff, level);
    unsigned int tileNr = TIFFComputeTile(_tiff, levelStartX, levelStartY, 0, 0);
    unsigned int total_tiles = TIFFNumberOfTiles(_tiff);
    if (tileNr < total_tiles) {
      unsigned long long* tbc = NULL;
      TIFFGetField(_tiff, TIFFTAG_TILEBYTECOUNTS, &tbc);
      unsigned long long k = tbc[tileNr];
      if (k == 0) {
        return -1;
      }
      unsigned int count = 0;
      unsigned char* jpt;
      if (TIFFGetField(_tiff, TIFFTAG_JPEGTABLES, &count, &jpt) != 0) {
        if (count > 4) {
          k = k + count;
          k -= 2; /* don't use EOI of header or SOI of tile */
        }
      }
      return k;
    }
    else {
      return -1;
    }
  }
  else {
    return -1;
  }
}

unsigned char* TIFFImage::readEncodedDataFromImage(const long long& startX, const long long& startY, const unsigned int& level) {
  if (_tiff) {
    unsigned int codec = 0;
    TIFFGetField(_tiff, TIFFTAG_COMPRESSION, &codec);
    if (codec == 7) { // New style JPEG
      long long levelStartX = std::floor(startX / getLevelDownsample(level) + 0.5);
      long long levelStartY = std::floor(startY / getLevelDownsample(level) + 0.5);
      TIFFSetDirectory(_tiff, level);
      unsigned int tileNr = TIFFComputeTile(_tiff, levelStartX, levelStartY, 0, 0);
      if (tileNr < TIFFNumberOfTiles(_tiff)) {
        unsigned char table_end[2];
        unsigned int count = 0;
        unsigned char* jpt;
        float* xfloatp;
        unsigned int endOfBuffer = 0;
        unsigned long long datasize = this->getEncodedTileSize(startX, startY, level);
        if (datasize < 0) {
          return NULL;
        }
        unsigned long long bufferoffset = 0;
        unsigned char* buffer = new unsigned char[datasize];

        if (TIFFGetField(_tiff, TIFFTAG_JPEGTABLES, &count, &jpt) != 0) {
          if (count > 4) {
            int retTIFFReadRawTile;
            /* Ignore EOI marker of JpegTables */
            _TIFFmemcpy(buffer, jpt, count - 2);
            bufferoffset += count - 2;
            /* Store last 2 bytes of the JpegTables */
            table_end[0] = buffer[bufferoffset - 2];
            table_end[1] = buffer[bufferoffset - 1];
            endOfBuffer = bufferoffset;
            bufferoffset -= 2;
            retTIFFReadRawTile = TIFFReadRawTile(
              _tiff,
              tileNr,
              (void*)&(((unsigned char*)buffer)[bufferoffset]),
              -1);
            bufferoffset += retTIFFReadRawTile;
            /* Overwrite SOI marker of image scan with previously */
            /* saved end of JpegTables */
            buffer[endOfBuffer - 2] = table_end[0];
            buffer[endOfBuffer - 1] = table_end[1];
          }
        }
        else {
          TIFFReadRawTile(
            _tiff,
            tileNr,
            (void*)&(((unsigned char*)buffer)[bufferoffset]),
            -1);
        }
        return buffer;
      }
      else {
        return NULL;
      }
    }
    else {
      return NULL;
    }
  }
  else {
    return NULL;
  }
}

template <typename T> T* TIFFImage::FillRequestedRegionFromTIFF(const long long& startX, const long long& startY, const unsigned long long& width,
  const unsigned long long& height, const unsigned int& level, unsigned int nrSamples)
{
  boost::shared_lock<boost::shared_mutex> l(*_openCloseMutex);
  T* temp = new T[width * height * nrSamples];
  std::fill(temp, temp + width * height * nrSamples, static_cast<T>(0));
  unsigned int tileW = _tileSizesPerLevel[level][0], tileH = _tileSizesPerLevel[level][1], levelH = _levelDimensions[level][1], levelW = _levelDimensions[level][0];

  long long levelStartX = std::floor(startX / getLevelDownsample(level) + 0.5);
  long long levelStartY = std::floor(startY / getLevelDownsample(level) + 0.5);
  long long startTileY = levelStartY - (levelStartY - ((levelStartY / tileH) * tileH));
  long long startTileX = levelStartX - (levelStartX - ((levelStartX / tileW) * tileW));
  long long finalX = levelStartX + width >= levelW ? levelW : levelStartX + width;
  long long finalY = levelStartY + height >= levelH ? levelH : levelStartY + height;

  for (long long iy = startTileY; iy < finalY; iy += tileH) {
    if (iy < 0) {
      continue;
    }
    for (long long ix = startTileX; ix < finalX; ix += tileW) {
      if (ix < 0) {
        continue;
      }

      std::stringstream k;
      k << ix * getLevelDownsample(level) << "-" << iy * getLevelDownsample(level) << "-" << level;
      bool deleteTile = false;
      unsigned int cachedTileSize = 0;
      T* tile = NULL;
      _cacheMutex->lock();
      std::static_pointer_cast<TileCache<T>>(_cache)->get(k.str(), tile, cachedTileSize);
      _cacheMutex->unlock();
      if (!tile) {
        tile = new T[tileW * tileH * getSamplesPerPixel()];
        std::fill(tile, tile + tileW * tileH * getSamplesPerPixel(), static_cast<T>(0.0));
        _cacheMutex->lock();
        TIFFSetDirectory(_tiff, level);
        unsigned int codec = 0;
        TIFFGetField(_tiff, TIFFTAG_COMPRESSION, &codec);
        unsigned int ycbcr = 0;
        TIFFGetField(_tiff, TIFFTAG_PHOTOMETRIC, &ycbcr);
        if (codec == 33005) {
          if (!_jp2000) {
            _jp2000 = new JPEG2000Codec();
          }
          unsigned int byteSize = tileW * tileH * getSamplesPerPixel() * sizeof(T);
          unsigned int rawSize = TIFFReadRawTile(_tiff, TIFFComputeTile(_tiff, ix, iy, 0, 0), tile, byteSize);
          _jp2000->decode((unsigned char*)tile, rawSize, byteSize);
        }
        else {
          if (codec == COMPRESSION_JPEG && ycbcr == PHOTOMETRIC_YCBCR) {
            TIFFSetField(_tiff, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);
          }
          TIFFReadTile(_tiff, tile, ix, iy, 0, 0);
          if (_colorType == pathology::RGBA) {
            for (unsigned int pos = 0; pos < tileW * tileH * nrSamples; pos += 4) {
              T b = tile[pos + 0];
              T g = tile[pos + 1];
              T r = tile[pos + 2];
              T a = tile[pos + 3];
              tile[pos + 0] = r;
              tile[pos + 1] = g;
              tile[pos + 2] = b;
              tile[pos + 3] = a;
            }
          }
        }
        if (std::static_pointer_cast<TileCache<T>>(_cache)->set(k.str(), tile, tileW * tileH * getSamplesPerPixel() * sizeof(T))) {
          deleteTile = true;
        }
        _cacheMutex->unlock();
      }

      long long ixx = (ix - levelStartX);
      long long iyy = (iy - levelStartY);
      long long lyh = levelStartY + height;
      long long lxw = levelStartX + width;
      long long ixw = ixx + tileW;
      long long rowLength = ixw > static_cast<long long>(width) ? (tileW - (ixw - width)) * nrSamples : tileW * nrSamples;
      long long tileDeltaX = 0;
      if (ixx < 0) {
        rowLength += ixx * nrSamples;
        tileDeltaX -= ixx * nrSamples;
        ixx = 0;
      }
      for (unsigned int ty = 0; ty < tileH; ++ty) {
        if ((iyy + ty >= 0) && (ixx >= 0) && (iyy + ty < static_cast<long long>(height)) && lxw > 0) {
          long long idx = (ty + iyy) * width * nrSamples + ixx * nrSamples;
          long long tids = (ty * tileW) * nrSamples;
          std::copy(tile + tids + tileDeltaX, tile + tids + rowLength + tileDeltaX, temp + idx);
        }
      }
      if (deleteTile) {
        delete[] tile;
      }
    }
  }
  return temp;
}