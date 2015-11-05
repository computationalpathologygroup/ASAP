#include "MultiResolutionImageWriter.h"
#include "MultiResolutionImage.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <math.h>
#include <algorithm>

extern "C" {
#include "tiffio.h"
};

#include "JPEG2000Codec.h"
#include "core/ProgressMonitor.h"

using namespace std;
using namespace pathology;

MultiResolutionImageWriter::MultiResolutionImageWriter() : _tiff(NULL),
  _codec(LZW), _quality(30),_tileSize(512), _pos(0), _numberOfIndexedColors(0), 
  _interpolation(pathology::Linear), _monitor(NULL), _cType(pathology::InvalidColorType),
  _dType(pathology::InvalidDataType), _min_vals(NULL), _max_vals(NULL)
{
  TIFFSetWarningHandler(NULL);
}

MultiResolutionImageWriter::~MultiResolutionImageWriter() {
  if (_tiff) {
    TIFFClose(_tiff);
    _tiff = NULL;
  }
}

void MultiResolutionImageWriter::setSpacing(std::vector<double>& spacing) {
  if (_tiff) {
    TIFFSetField(_tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
    if (spacing.empty()) {
      double defaultSpacing = 1.0;
      TIFFSetField(_tiff, TIFFTAG_XRESOLUTION, defaultSpacing);
      TIFFSetField(_tiff, TIFFTAG_YRESOLUTION, defaultSpacing);
    }
    else {
      double pixPerCmX = (1. / spacing[0]) * 10000;
      double pixPerCmY = (1. / spacing[1]) * 10000;
      TIFFSetField(_tiff, TIFFTAG_XRESOLUTION, pixPerCmX);
      TIFFSetField(_tiff, TIFFTAG_YRESOLUTION, pixPerCmY);
    }
  }
}

void MultiResolutionImageWriter::setProgressMonitor(ProgressMonitor* monitor) {
  _monitor = monitor;
}

void MultiResolutionImageWriter::writeImageToFile(MultiResolutionImage* img, const std::string& fileName) {
  setColorType(img->getColorType());
  setDataType(img->getDataType());
  unsigned int cDepth = 1;
  if (_cType == RGB) {
    cDepth = 3;
  } else if (_cType == ARGB) {
    cDepth = 4;
  } else if (_cType == Indexed) {    
    cDepth = img->getSamplesPerPixel();
    setNumberOfIndexedColors(cDepth);
  }
  unsigned int nrBits = 8;
  if (_dType == UInt32 || _dType == Float) {
    nrBits = 32;
  } else if (_dType == UInt16) {
    nrBits = 16;
  }
  if (openFile(fileName) == 0) {
    std::vector<unsigned long long> dims = img->getDimensions();
    std::vector<double> spacing = img->getSpacing();
    // Check whether spacing override has been added
    if (!_overrideSpacing.empty()) {
      spacing = _overrideSpacing;
    }
    setSpacing(spacing);
    if (writeImageInformation(dims[0], dims[1]) == 0) {
      for (int y =0; y < dims[1]; y+=_tileSize) {
        for (int x =0; x < dims[0]; x+=_tileSize) {
          unsigned char* data = new unsigned char[_tileSize*_tileSize*cDepth*(nrBits/8)];
          if (_dType == pathology::UInt32) {
            img->getRawRegion(x,y,_tileSize,_tileSize,0,(unsigned int*&)data);
          }
          else if (_dType == pathology::UInt16) {
            img->getRawRegion(x,y,_tileSize,_tileSize,0,(unsigned short*&)data);
          }
          else if (_dType == pathology::Float) {
            img->getRawRegion(x,y,_tileSize,_tileSize,0,(float*&)data);
          }
          else if (_dType == pathology::UChar) {
            img->getRawRegion(x,y,_tileSize,_tileSize,0,data);            
          }
          writeBaseImagePart((void*)data);
          delete[] data;
          data = NULL;
        }
      }
      finishImage();
    } else {
      cerr << "Could not write image information" << endl;
    }
  } else{
    cerr << "Failed to open TIFF file for writing" << endl;
  }
}

int MultiResolutionImageWriter::openFile(const std::string& fileName) {
  _tiff = TIFFOpen(fileName.c_str(), "w8");  
  if (!_tiff) {
    cerr << "Failed to open TIFF file for writing" << endl;
    return -1;
  }
  _fileName = fileName;
  _pos = 0;
  _levelFiles.clear();
  return 0;
}

int MultiResolutionImageWriter::writeImageInformation(const unsigned long long& sizeX, const unsigned long long& sizeY) {
  if (_tiff) {
    unsigned int cDepth = 1;
    if (_cType == RGB) {
      cDepth = 3;
    }
    else if (_cType == ARGB) {
      cDepth = 4;
    }
    else if (_cType == Indexed) {
      cDepth = getNumberOfIndexedColors();
    }
    _min_vals = new double[cDepth];
    _max_vals = new double[cDepth];
    for (int i = 0; i < cDepth; ++i) {
      _min_vals[i] = std::numeric_limits<double>::max();
      _max_vals[i] = std::numeric_limits<double>::min();
    }
    setPyramidTags(_tiff, sizeX, sizeY);
    unsigned int totalSteps = (sizeX * sizeY) / (_tileSize * _tileSize);
    if (_monitor) {
      _monitor->setMaximumProgress(2 * totalSteps);
      _monitor->setProgress(0);
    }
    return 0;
  } else {
    return -1;
  }
}

void MultiResolutionImageWriter::writeBaseImagePart(void* data) {
  writeBaseImagePartToTIFFTile(data, _pos);
  ++_pos;
}

void MultiResolutionImageWriter::writeBaseImagePartToLocation(void* data, const unsigned long long& x, const unsigned long long& y) {
  unsigned int pos = TIFFComputeTile(_tiff, x, y, 0, 0);
  writeBaseImagePartToTIFFTile(data, pos);
}

void MultiResolutionImageWriter::writeBaseImagePartToTIFFTile(void* data, unsigned int pos) {
  unsigned int cDepth = 1;
  if (_cType == RGB) {
    cDepth = 3;
  }
  else if (_cType == ARGB) {
    cDepth = 4;
  }
  else if (_cType == Indexed) {
    if (_numberOfIndexedColors == 0) {
      return;
    }
    cDepth = _numberOfIndexedColors;
  }
  unsigned int npixels = _tileSize * _tileSize * cDepth;

  //Determine min/max of tile part
  if (_dType == pathology::UInt32) {
    unsigned int *temp = (unsigned int*)data;
    for (int i = 0; i < _tileSize*_tileSize*cDepth; i += cDepth) {
      for (int j = 0; j < cDepth; ++j) {
        double val = temp[i + j];
        if (val > _max_vals[j]) {
          _max_vals[j] = val;
        }
        if (val < _min_vals[j]) {
          _min_vals[j] = val;
        }
      }
    }
  }
  else if (_dType == pathology::UInt16) {
    unsigned short *temp = (unsigned short*)data;
    for (int i = 0; i < _tileSize*_tileSize*cDepth; i += cDepth) {
      for (int j = 0; j < cDepth; ++j) {
        double val = temp[i + j];
        if (val > _max_vals[j]) {
          _max_vals[j] = val;
        }
        if (val < _min_vals[j]) {
          _min_vals[j] = val;
        }
      }
    }
  }
  else if (_dType == pathology::Float) {
    float *temp = (float*)data;
    for (int i = 0; i < _tileSize*_tileSize*cDepth; i += cDepth) {
      for (int j = 0; j < cDepth; ++j) {
        double val = temp[i + j];
        if (val > _max_vals[j]) {
          _max_vals[j] = val;
        }
        if (val < _min_vals[j]) {
          _min_vals[j] = val;
        }
      }
    }
  }
  else if (_dType == pathology::UChar) {
    unsigned char *temp = (unsigned char*)data;
    for (int i = 0; i < _tileSize*_tileSize*cDepth; i += cDepth) {
      for (int j = 0; j < cDepth; ++j) {
        double val = temp[i + j];
        if (val > _max_vals[j]) {
          _max_vals[j] = val;
        }
        if (val < _min_vals[j]) {
          _min_vals[j] = val;
        }
      }
    }
  }

  if (getCompression() == JPEG2000_LOSSLESS || getCompression() == JPEG2000_LOSSY) {
    int depth = 8;
    unsigned int size = npixels * sizeof(unsigned char);
    if (getDataType() == UInt32 || getDataType() == Float) {
      depth = 32;
      size = npixels * sizeof(float);
    }
    else if (getDataType() == UInt16) {
      depth = 16;
      size = npixels * sizeof(unsigned short);
    }

    unsigned int nrComponents = 3;
    if (getColorType() == ARGB) {
      nrComponents = 4;
    }
    else if (getColorType() == Monochrome) {
      nrComponents = 1;
    }
    else if (getColorType() == Indexed) {
      nrComponents = _numberOfIndexedColors;
    }

    float rate = 1.0;
    if (getCompression() == JPEG2000_LOSSY) {
      rate = getJPEGQuality() / 100.;
    }
    JPEG2000Codec cod;
    cod.encode((char*)data, size, _tileSize, depth, nrComponents, rate);
    TIFFWriteRawTile(_tiff, pos, data, size);
  }
  else {
    if (_dType == Float) {
      TIFFWriteEncodedTile(_tiff, pos, data, npixels * sizeof(float));
    }
    else if (_dType == UInt16) {
      TIFFWriteEncodedTile(_tiff, pos, data, npixels * sizeof(unsigned short));
    }
    else if (_dType == UInt32) {
      TIFFWriteEncodedTile(_tiff, pos, data, npixels * sizeof(unsigned int));
    }
    else {
      TIFFWriteEncodedTile(_tiff, pos, data, npixels * sizeof(unsigned char));
    }
  }
  if (_monitor) {
    ++(*_monitor);
  }
}

int MultiResolutionImageWriter::finishImage() {
  TIFFSetField(_tiff, TIFFTAG_PERSAMPLE, PERSAMPLE_MULTI);
  TIFFSetField(_tiff, TIFFTAG_SMINSAMPLEVALUE, &_min_vals[0]);
  TIFFSetField(_tiff, TIFFTAG_SMAXSAMPLEVALUE, &_max_vals[0]);
  /* Reset to default behavior, if needed. */
  TIFFSetField(_tiff, TIFFTAG_PERSAMPLE, PERSAMPLE_MERGED);
  delete[] _min_vals;
  delete[] _max_vals;
  _min_vals = NULL;
  _max_vals = NULL;
  if (getDataType() == UInt32) {
    writePyramidToDisk<unsigned int>();
    incorporatePyramid<unsigned int>();
  }
  else if (getDataType() == UInt16) {
    writePyramidToDisk<unsigned short>();
    incorporatePyramid<unsigned short>();
  }
  else if (getDataType() == UChar) {
    writePyramidToDisk<unsigned char>();
    incorporatePyramid<unsigned char>();
  }
  else {
    writePyramidToDisk<float>();
    incorporatePyramid<float>();
  }
  for (std::vector<std::string>::const_iterator it = _levelFiles.begin(); it != _levelFiles.end(); ++it) {
    for (int i = 0; i < 5; ++i) {
      if (remove(it->c_str()) == 0) {
        break;
      }
    }
  }
  TIFFClose(_tiff);
  _tiff = NULL;
  _levelFiles.clear();
  _fileName = "";
  _pos = 0;
  return 0;
}

template <typename T> int MultiResolutionImageWriter::writePyramidToDisk() {

  //! First get the overall image width and height;
  unsigned long w = 0, h = 0, nrsamples=0, nrbits=0;
  // TIFF idiosyncracy, when setting resolution tags one uses doubles,
  // getting them requires floats
  float spacingX = 0, spacingY = 0;
  std::vector<double> spacing;
  TIFFGetField(_tiff, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(_tiff, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetField(_tiff, TIFFTAG_SAMPLESPERPIXEL, &nrsamples);
  TIFFGetField(_tiff, TIFFTAG_BITSPERSAMPLE, &nrbits);
  TIFFGetField(_tiff, TIFFTAG_XRESOLUTION, &spacingX);
  TIFFGetField(_tiff, TIFFTAG_YRESOLUTION, &spacingY);
  spacing.push_back(1. / (spacingX / (10000.)));
  spacing.push_back(1. / (spacingY / (10000.)));
  // Determine the amount of pyramid levels
  unsigned int pyramidlevels = 1;
  unsigned int lowestwidth = w;
  while (lowestwidth > 1024) {
    lowestwidth /= 2;
    pyramidlevels += 1;
  }
  if (abs(1024. - lowestwidth) > abs(1024. - lowestwidth*2)) {
    lowestwidth *= 2;
    pyramidlevels -= 1;
  }
  // Setup the image directory for the thumbnail
  unsigned int lowestheight = (unsigned int)(h/pow(2.0,(double)pyramidlevels));

  // Write temporary image to store previous level (LibTiff does not allow to go back and forth between
  // empty directories
#ifdef WIN32
  size_t found = _fileName.find_last_of("/\\");
#else 
  size_t found = _fileName.find_last_of("/");
#endif
  string tmpPth = _fileName.substr(0,found+1);
  string fileName = _fileName.substr(found+1);
  size_t dotLoc = fileName.find_last_of(".");
  string baseName = fileName.substr(0,dotLoc);
  for (unsigned int level = 1; level <= pyramidlevels; ++level) {
    if (_monitor) {
      _monitor->setProgress((_monitor->maximumProgress() / 2.) + (static_cast<float>(level) / static_cast<float>(pyramidlevels))* (_monitor->maximumProgress() / 4.));
    }
    TIFF* prevLevelTiff = _tiff;
    if (level != 1) {
      std::stringstream ssm;
      ssm << tmpPth << "temp" << baseName << "Level" << level-1 << ".tif";
      prevLevelTiff = TIFFOpen(ssm.str().c_str(),"r");
    }
    std::stringstream ssm;
    ssm << tmpPth << "temp" << baseName << "Level" << level << ".tif";
    TIFF* levelTiff = TIFFOpen(ssm.str().c_str(),"w8");
    _levelFiles.push_back(ssm.str());
    unsigned int levelw = (unsigned int)(w/pow(2.,(double)level));
    unsigned int levelh = (unsigned int)(h/pow(2.,(double)level));
    unsigned int prevLevelw = (unsigned int)(w/pow(2.,(double)level-1));
    unsigned int prevLevelh = (unsigned int)(h/pow(2.,(double)level-1));
    setTempPyramidTags(levelTiff, levelw, levelh);
    unsigned int prevLevelTiles = TIFFNumberOfTiles(prevLevelTiff);
    unsigned int nrTilesX = (unsigned int)ceil(float(levelw)/_tileSize);
    unsigned int nrTilesY = (unsigned int)ceil(float(levelh)/_tileSize);
    unsigned int levelTiles = nrTilesX*nrTilesY;
    unsigned int npixels = _tileSize*_tileSize*nrsamples;
    int rowOrg=-2, colOrg=0;
    for (unsigned int i  = 0; i < levelTiles;++i) {          
      if (i%nrTilesX==0) {
        rowOrg += 2;
        colOrg = 0;
      }
      unsigned int xpos = _tileSize * colOrg;
      unsigned int ypos = _tileSize * rowOrg;
      T* tile1 = (T*) _TIFFmalloc(npixels * sizeof(T));
      T* tile2 = (T*) _TIFFmalloc(npixels * sizeof(T));
      T* tile3 = (T*) _TIFFmalloc(npixels * sizeof(T));
      T* tile4 = (T*) _TIFFmalloc(npixels * sizeof(T));
      T* outTile = (T*) _TIFFmalloc(npixels * sizeof(T));
      unsigned int size =  npixels * sizeof(T);
      JPEG2000Codec cod;
      if (level == 1 && (getCompression() == JPEG2000_LOSSY || getCompression() == JPEG2000_LOSSLESS)) {
        int tileNr = TIFFComputeTile(prevLevelTiff, xpos, ypos, 0, 0);
        unsigned int outTileSize = _tileSize*_tileSize*nrsamples*(nrbits/8);
        unsigned int rawSize = TIFFReadRawTile(prevLevelTiff, tileNr, tile1, outTileSize);
        cod.decode((char*)tile1, rawSize, outTileSize);
        if (xpos+_tileSize>=prevLevelw) {
          std::fill_n(tile2, npixels, 0);
        } else {
          tileNr = TIFFComputeTile(prevLevelTiff, xpos+_tileSize, ypos, 0, 0);
          unsigned int rawSize = TIFFReadRawTile(prevLevelTiff, tileNr, tile2, outTileSize);
          cod.decode((char*)tile2, rawSize, outTileSize);
        }
        if (ypos+_tileSize>=prevLevelh) {
          std::fill_n(tile3, npixels, 0);
        } else {
          tileNr = TIFFComputeTile(prevLevelTiff, xpos, ypos+_tileSize, 0, 0);
          unsigned int rawSize = TIFFReadRawTile(prevLevelTiff, tileNr, tile3, outTileSize);
          cod.decode((char*)tile3, rawSize, outTileSize);
        }
        if (xpos+_tileSize>=prevLevelw || ypos+_tileSize>=prevLevelh) {
          std::fill_n(tile4, npixels, 0);
        } else {
          tileNr = TIFFComputeTile(prevLevelTiff, xpos+_tileSize, ypos+_tileSize, 0, 0);
          unsigned int rawSize = TIFFReadRawTile(prevLevelTiff, tileNr, tile4, outTileSize);
          cod.decode((char*)tile4, rawSize, outTileSize);
        }
      } else {
        TIFFReadTile(prevLevelTiff, tile1, xpos, ypos, 0, 0);
        if (xpos+_tileSize>=prevLevelw) {
          std::fill_n(tile2, npixels, 0);
        } else {
          TIFFReadTile(prevLevelTiff, tile2, xpos+_tileSize, ypos, 0, 0);
        }
        if (ypos+_tileSize>=prevLevelh) {
          std::fill_n(tile3, npixels, 0);
        } else {
          TIFFReadTile(prevLevelTiff, tile3, xpos, ypos+_tileSize, 0, 0);
        }
        if (xpos+_tileSize>=prevLevelw || ypos+_tileSize>=prevLevelh) {
          std::fill_n(tile4, npixels, 0);
        } else {
          TIFFReadTile(prevLevelTiff, tile4, xpos+_tileSize, ypos+_tileSize, 0, 0);
        }
      }
      T* dsTile1 = downscaleTile(tile1, _tileSize, nrsamples);
      T* dsTile2 = downscaleTile(tile2, _tileSize, nrsamples);
      T* dsTile3 = downscaleTile(tile3, _tileSize, nrsamples);
      T* dsTile4 = downscaleTile(tile4, _tileSize, nrsamples);
      unsigned int dsSize = _tileSize/2;
      for (unsigned int y = 0; y < _tileSize; ++y) {
        for (unsigned int x = 0; x < _tileSize; ++x) {
          for (unsigned int s = 0; s < nrsamples; ++s) {
            unsigned int outIndex = nrsamples*(y*_tileSize + x) + s;
            T* usedTile = dsTile1;
            unsigned int inIndex = y*dsSize*nrsamples + x*nrsamples +s;
            if (x >= dsSize && y < dsSize) {
              usedTile = dsTile2;
              inIndex = y*dsSize*nrsamples + ((x-dsSize)*nrsamples) + s;
            }
            else if (x < dsSize && y >= dsSize) {
              usedTile = dsTile3;
              inIndex = (y-dsSize)*dsSize*nrsamples + x*nrsamples + s;
            }
            else if (x >= dsSize && y >= dsSize) {
              usedTile = dsTile4;
              inIndex = (y-dsSize)*dsSize*nrsamples + (x-dsSize)*nrsamples + s;
            }
            T val = *(usedTile+inIndex);
            *(outTile+outIndex) = val;
          }
        }
      }
      TIFFWriteEncodedTile(levelTiff, i, outTile, npixels * sizeof(T));
      _TIFFfree(tile1);_TIFFfree(dsTile1);
      _TIFFfree(tile2);_TIFFfree(dsTile2);
      _TIFFfree(tile3);_TIFFfree(dsTile3);
      _TIFFfree(tile4);_TIFFfree(dsTile4);
      _TIFFfree(outTile);
      colOrg += 2;
    }
    if (level != 1) {
      TIFFClose(prevLevelTiff);
    }
    spacing[0] *= 2.;
    spacing[1] *= 2.;
    TIFFSetField(_tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
    double pixPerCmX = (1. / spacing[0]) * 10000;
    double pixPerCmY = (1. / spacing[1]) * 10000;
    TIFFSetField(levelTiff, TIFFTAG_XRESOLUTION, pixPerCmX);
    TIFFSetField(levelTiff, TIFFTAG_YRESOLUTION, pixPerCmY);
    TIFFClose(levelTiff);
  }
  //! Write base directory to disk
  TIFFWriteDirectory(_tiff);
  return 0;
}

template <typename T> int MultiResolutionImageWriter::incorporatePyramid() {
  unsigned long nrsamples = 0;

  //// Now add all the pyramid levels
  for (vector<string>::const_iterator it = _levelFiles.begin(); it != _levelFiles.end(); ++it) {
    if (_monitor) {
      _monitor->setProgress(3 * (_monitor->maximumProgress() / 4.) + ((static_cast<float>(it - _levelFiles.begin()) + 1.0) / static_cast<float>(_levelFiles.size()))* (_monitor->maximumProgress() / 4.));
    }
    TIFF * level = TIFFOpen(it->c_str(), "rm");

    float spacingX = 0, spacingY = 0;
    std::vector<double> spacing;
    TIFFGetField(level, TIFFTAG_XRESOLUTION, &spacingX);
    TIFFGetField(level, TIFFTAG_YRESOLUTION, &spacingY);
    spacing.push_back(1. / (spacingX / (10000.)));
    spacing.push_back(1. / (spacingY / (10000.)));

    unsigned int levelw, levelh;
    TIFFGetField(level, TIFFTAG_IMAGEWIDTH, &levelw);
    TIFFGetField(level, TIFFTAG_IMAGELENGTH, &levelh);
    setPyramidTags(_tiff, levelw, levelh);
    TIFFGetField(level, TIFFTAG_SAMPLESPERPIXEL, &nrsamples);
    writePyramidLevel<T>(level, levelw, levelh, nrsamples);

    setSpacing(spacing);
    TIFFWriteDirectory(_tiff);
    TIFFClose(level);
  }
  return 0;
}

template int MultiResolutionImageWriter::incorporatePyramid<unsigned int>();

void MultiResolutionImageWriter::setBaseTags(TIFF* levelTiff) {
  TIFFSetField(levelTiff, TIFFTAG_SUBFILETYPE, 0);
  if (_cType == Monochrome || _cType == Indexed) {
    TIFFSetField(levelTiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  } else if (_cType == ARGB || _cType == RGB) {
    TIFFSetField(levelTiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  } 

  if (_dType == UChar) {
    TIFFSetField(levelTiff, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(levelTiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
  } else if (_dType == UInt32) {
    TIFFSetField(levelTiff, TIFFTAG_BITSPERSAMPLE, sizeof(unsigned int)*8);
    TIFFSetField(levelTiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
  } else if (_dType == UInt16) {
    TIFFSetField(levelTiff, TIFFTAG_BITSPERSAMPLE, sizeof(unsigned short)*8);
    TIFFSetField(levelTiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
  } else if (_dType == Float) {
    TIFFSetField(levelTiff, TIFFTAG_BITSPERSAMPLE, sizeof(float)*8);
    TIFFSetField(levelTiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  }
  if (_cType == Monochrome) {
    TIFFSetField(levelTiff, TIFFTAG_SAMPLESPERPIXEL, 1);
  } else if (_cType == RGB) {
    TIFFSetField(levelTiff, TIFFTAG_SAMPLESPERPIXEL, 3);
  } else if (_cType == ARGB) {
    TIFFSetField(levelTiff, TIFFTAG_SAMPLESPERPIXEL, 4);
  } else if (_cType == Indexed) {
    TIFFSetField(levelTiff, TIFFTAG_SAMPLESPERPIXEL, _numberOfIndexedColors);
  }
  TIFFSetField(levelTiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(levelTiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
}

void MultiResolutionImageWriter::setPyramidTags(TIFF* levelTiff, const unsigned long long& width, const unsigned long long& hight) {
  setBaseTags(levelTiff);
  if (_codec == LZW) {
    TIFFSetField(levelTiff, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
  } else if (_codec == JPEG) {
    TIFFSetField(levelTiff, TIFFTAG_COMPRESSION, COMPRESSION_JPEG);
    TIFFSetField(levelTiff, TIFFTAG_JPEGQUALITY, (unsigned int)_quality);
  } else if (_codec == RAW) {
    TIFFSetField(levelTiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  }
  else if (_codec == JPEG2000_LOSSLESS || _codec == JPEG2000_LOSSY) {
    TIFFSetField(levelTiff, TIFFTAG_COMPRESSION, 33005);
  }

  TIFFSetField(levelTiff, TIFFTAG_TILEWIDTH, _tileSize);
  TIFFSetField(levelTiff, TIFFTAG_TILELENGTH, _tileSize);
  TIFFSetField(levelTiff, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(levelTiff, TIFFTAG_IMAGELENGTH, hight); 
}

void MultiResolutionImageWriter::setTempPyramidTags(TIFF* levelTiff, const unsigned long long& width, const unsigned long long& hight) {
  setBaseTags(levelTiff);
  TIFFSetField(levelTiff, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
  TIFFSetField(levelTiff, TIFFTAG_TILEWIDTH, _tileSize);
  TIFFSetField(levelTiff, TIFFTAG_TILELENGTH, _tileSize);
  TIFFSetField(levelTiff, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(levelTiff, TIFFTAG_IMAGELENGTH, hight); 
}

template <typename T> T* MultiResolutionImageWriter::downscaleTile(T* inTile, unsigned int tileSize, unsigned int nrSamples) {  
  unsigned int dsSize = tileSize/2;
  unsigned int npixels = dsSize*dsSize*nrSamples;
  T* dsTile = (T*) _TIFFmalloc(dsSize*dsSize * nrSamples * sizeof(T));
  int rowOrg=0, colOrg=0;
  for (unsigned long long y = 0; y < dsSize; ++y) {
    for (unsigned long long x = 0; x < dsSize; ++x) {
      for (unsigned long long s = 0; s < nrSamples; ++s) {
        unsigned int index = (2*y*tileSize*nrSamples)+(2*x*nrSamples) + s;
        unsigned int dsIndex = (y*dsSize*nrSamples)+(x*nrSamples) + s;
        if (_interpolation == pathology::Linear) {
          *(dsTile+dsIndex) = (T)(*(inTile+index)/4. + *(inTile+index+tileSize*nrSamples)/4. + *(inTile+index+nrSamples)/4. + *(inTile+index+tileSize*nrSamples+nrSamples)/4.);
        } else {
          *(dsTile+dsIndex) = (T)(*(inTile+index));
        }
      }
    }
  }
  return dsTile;
}

template <typename T> void MultiResolutionImageWriter::writePyramidLevel(TIFF* levelTiff, unsigned int levelwidth, unsigned int levelheight, unsigned int nrsamples) {
	unsigned int npixels = _tileSize * _tileSize * nrsamples;
	T* raster = (T*) _TIFFmalloc(npixels * sizeof (T));
	if (getCompression() == JPEG2000_LOSSLESS || getCompression() == JPEG2000_LOSSY) {
	  JPEG2000Codec cod = JPEG2000Codec();
	  int depth = 8;
	  unsigned int size = npixels * sizeof(unsigned char);
	  if (getDataType() == UInt32 || getDataType() == Float) {
		  depth = 32;
		  size = npixels * sizeof(T);
	  }

	  unsigned int nrComponents = 3;
	  if (getColorType()==ARGB) {
		  nrComponents = 4;
	  } else if (getColorType()==Monochrome) {
		  nrComponents = 1;
	  } else if (getColorType()==Indexed) {
		  nrComponents = _numberOfIndexedColors;
	  }

	  float rate = 1.0;
	  if (getCompression() == JPEG2000_LOSSY) {
		  rate = getJPEGQuality()/100.;
	  }
	  for (unsigned int i = 0; i < TIFFNumberOfTiles(levelTiff); ++i) {
      TIFFReadEncodedTile(levelTiff, i, raster, npixels * sizeof(T));
		  unsigned int size = npixels * sizeof(T);
		  cod.encode((char*)raster, size, _tileSize, depth, nrComponents, rate);
		  TIFFWriteRawTile(_tiff, i, raster, size);
	  }
  } else {
		for (unsigned int i = 0; i < TIFFNumberOfTiles(levelTiff); ++i) {
			TIFFReadEncodedTile(levelTiff, i, raster, npixels * sizeof(T));
			TIFFWriteEncodedTile(_tiff, i, raster, npixels * sizeof(T));
		}
	}
	_TIFFfree(raster);
}