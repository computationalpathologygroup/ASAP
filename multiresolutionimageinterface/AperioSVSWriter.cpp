#include "AperioSVSWriter.h"
#include "JPEG2000Codec.h"
#include <sstream>
#include "core/PathologyEnums.h"

extern "C" {
#include "tiffio.h"
};

using namespace pathology;

int AperioSVSWriter::finishImage() {
  if (getDataType() == UInt32) {
    writePyramidToDisk<unsigned int>();
    writeThumbnail<unsigned int>();
    incorporatePyramid<unsigned int>();
    writeThumbnail<unsigned int>();
  }
  else if (getDataType() == UInt16) {
    writePyramidToDisk<unsigned short>();
    writeThumbnail<unsigned short>();
    incorporatePyramid<unsigned short>();
    writeThumbnail<unsigned short>();
  }
  else if (getDataType() == UChar) {
    writePyramidToDisk<unsigned char>();
    writeThumbnail<unsigned char>();
    incorporatePyramid<unsigned char>();
    writeThumbnail<unsigned char>();
  }
  else {
    writePyramidToDisk<float>();
    writeThumbnail<float>();
    incorporatePyramid<float>();
    writeThumbnail<float>();
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

template <typename T>
void AperioSVSWriter::writeThumbnail() {
  unsigned long w = 0, h = 0, nrsamples = 0;
  TIFF* lowestResTiff = TIFFOpen(_levelFiles[_levelFiles.size()-1].c_str(), "r");
  TIFFGetField(lowestResTiff, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(lowestResTiff, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetField(lowestResTiff, TIFFTAG_SAMPLESPERPIXEL, &nrsamples);

  setBaseTags(_tiff);
  TIFFSetField(_tiff, TIFFTAG_COMPRESSION, COMPRESSION_JPEG);
  TIFFSetField(_tiff, TIFFTAG_JPEGQUALITY, 70);
  TIFFSetField(_tiff, TIFFTAG_ROWSPERSTRIP, 16);
  TIFFSetField(_tiff, TIFFTAG_IMAGEWIDTH, w);
  TIFFSetField(_tiff, TIFFTAG_IMAGELENGTH, h);
  TIFFSetField(_tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
  TIFFSetField(_tiff, TIFFTAG_SUBFILETYPE, FILETYPE_REDUCEDIMAGE);

  unsigned int npixels = w * h * nrsamples;
  T* raster = (T*)_TIFFmalloc(npixels * sizeof(T));
  unsigned int nrTiles = TIFFNumberOfTiles(lowestResTiff);
  unsigned int tileW = 0, tileH = 0;
  TIFFGetField(lowestResTiff, TIFFTAG_TILEWIDTH, &tileW);
  TIFFGetField(lowestResTiff, TIFFTAG_TILELENGTH, &tileH);
  T* tile = (T*)_TIFFmalloc(tileW * tileH * nrsamples * sizeof(T));
  JPEG2000Codec cod;
  for (unsigned int tileY = 0; tileY < h; tileY += tileH) {
    for (unsigned int tileX = 0; tileX < w; tileX += tileW) {
      if (getCompression() == JPEG2000) {
        unsigned int no = TIFFComputeTile(lowestResTiff, tileX, tileY, 0, 0);
        unsigned int rawSize = TIFFReadRawTile(lowestResTiff, no, tile, tileW*tileH*nrsamples*sizeof(T));
        cod.decode((unsigned char*)tile, rawSize, tileW*tileH*nrsamples*sizeof(T));
      }
      else {
        TIFFReadTile(lowestResTiff, tile, tileX, tileY, 0, 0);
      }
      for (unsigned int y = 0; y < tileH; ++y){
        for (unsigned int x = 0; x < tileW; ++x){
          if ((tileX + x) < w && (tileY + y) < h) {
            for (unsigned int s = 0; s < nrsamples; ++s) {
              (*(raster + (tileY + y) * w * nrsamples + (tileX + x) * nrsamples + s)) = (*(tile + y * tileW * nrsamples + x * nrsamples + s));
            }
          }
        }
      }
    }
  }
  _TIFFfree(tile);
  tile = NULL;
  if (raster != NULL) {
    unsigned int nrOfStrips = TIFFNumberOfStrips(_tiff);
    for (unsigned int i = 0; i < nrOfStrips; ++i) {
      T* strip = (T*)_TIFFmalloc(16 * w * nrsamples * sizeof(T));
      unsigned int startOfStrip = i * 16 * w * nrsamples;
      unsigned int endOfStrip = (i + 1) * 16 * w * nrsamples;
      if (endOfStrip>npixels) {
        endOfStrip = npixels;
      }
      for (unsigned int j = startOfStrip, k = 0; j < endOfStrip; ++j, ++k) {
        T val = *(raster + j);
        *(strip + k) = val;
      }
      TIFFWriteEncodedStrip(_tiff, i, strip, 16 * w * nrsamples * sizeof(T));
      _TIFFfree(strip);
    }
    _TIFFfree(raster);
  }

  TIFFClose(lowestResTiff);
  lowestResTiff = NULL;
  if (TIFFNumberOfDirectories(_tiff) > 1) {
    TIFFSetField(_tiff, TIFFTAG_IMAGEDESCRIPTION, "Aperio Image macro");
  }
  TIFFWriteDirectory(_tiff);
}

void AperioSVSWriter::setSpacing(std::vector<double>& spacing) {
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
    std::stringstream ssm;
    ssm << "Aperio Image|MPP =";
    ssm << spacing[0];
    std::string test = ssm.str();
    TIFFSetField(_tiff, TIFFTAG_IMAGEDESCRIPTION, test.c_str());
  }
}