#include "VSIImage.h"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "core/filetools.h"
#include "core/Box.h"
#include "core/PathologyEnums.h"

// Include DCMTK LIBJPEG for lossy and lossless JPEG compression
extern "C" {
#define boolean ijg_boolean
#include "dcmtk/dcmjpeg/libijg8/jpeglib8.h"
#include "jpeg_mem_src.h"
#undef boolean
#undef const
}

#include "JPEG2000Codec.h"

using namespace pathology;
using namespace std;

VSIImage::VSIImage() : MultiResolutionImage(),
	_vsiFileName(""), _etsFile(""), _tileOffsets(), _tileCoords(),
	_tileSizeX(0), _tileSizeY(0), _nrTilesX(0),
	_nrTilesY(0), _compressionType(0) {
}

VSIImage::~VSIImage() {
  cleanup();
}

void VSIImage::cleanup() {
	_vsiFileName = "";
	_etsFile = "";
	_tileCoords.clear();
	_tileOffsets.clear();
	_tileSizeX = 0;
	_tileSizeY = 0;
	_nrTilesX = 0;
	_nrTilesY = 0;
  _compressionType = 0;
  MultiResolutionImage::cleanup();
}

bool VSIImage::initializeType(const std::string& imagePath) {
	cleanup();
	if (!core::fileExists(imagePath)) {
		return false;
	}
	_vsiFileName = imagePath;
  string pth = core::extractFilePath(imagePath);
	string fileName = core::extractFileName(imagePath);	
  string baseName = core::extractBaseName(imagePath);    
	vector<string> etsFiles;
	core::getFiles(core::completePath("_" + baseName + "_",pth),"*.ets",etsFiles, true);
  if (etsFiles.empty()) {
    core::getFiles(core::completePath(baseName, pth), "*.ets", etsFiles, true);
    if (etsFiles.empty()) {
      core::getFiles("_" + core::completePath(baseName, pth), "*.ets", etsFiles, true);
      if (etsFiles.empty()) {
        core::getFiles(core::completePath(baseName, pth) + "_", "*.ets", etsFiles, true);
        if (etsFiles.empty()) {
          std::cout << "Could not find the .ets files belonging to " << baseName << ".vsi, is the VSI-folder present and correctly named?" << std::endl;
          return false;
        }
      }
    }
  }
  unsigned long long _mostNrPixels = 0;
	for (unsigned int i = 0; i < etsFiles.size(); ++i) {
		ifstream ets;
		ets.open(etsFiles[i].c_str(), ios::in | ios::binary);
		if (ets.good()) {
      unsigned long long nrPixels = parseETSFile(ets);
      if (nrPixels > _mostNrPixels) {
        _mostNrPixels = nrPixels;
        _etsFile = etsFiles[i];
      }
		}
		ets.close();	
	  _tileCoords.clear();
	  _tileOffsets.clear();
	  _tileSizeX = 0;
	  _tileSizeY = 0;
	  _nrTilesX = 0;
	  _nrTilesY = 0;
    _compressionType = 0;
    _levelDimensions.clear();
	}
	ifstream ets;
	ets.open(_etsFile.c_str(), ios::in | ios::binary);
  parseETSFile(ets);
  ets.close();	
  _fileType = "vsi";
  return _isValid;
}

unsigned long long VSIImage::parseETSFile(std::ifstream& ets) {
  // Read general file info
  char* memblock = new char[4];
  char* memblockLong = new char[8];
  ets.read(memblock,4);
  ets.read(memblock,4);
  int headerSize = *reinterpret_cast<int*>(memblock);
  ets.read(memblock,4);
  int version = *reinterpret_cast<int*>(memblock);
  ets.read(memblock,4);
  int nDims = *reinterpret_cast<int*>(memblock);
  ets.read(memblockLong,8);
  long additionalHeaderOffset = *reinterpret_cast<long*>(memblockLong);
  ets.read(memblock,4);
  int additionalHeaderSize =  *reinterpret_cast<int*>(memblock);
  ets.seekg(4, ios::cur);
  ets.read(memblockLong, 8);
  unsigned long long usedChunkOffset = *reinterpret_cast<unsigned long long *>(memblockLong);
  ets.read(memblock,4);
  int nUsedChunks =  *reinterpret_cast<int*>(memblock);
  ets.seekg(additionalHeaderOffset);
  ets.read(memblock,4);
  ets.seekg(4, ios::cur);
  ets.read(memblock,4);
  int pixelType = *reinterpret_cast<int*>(memblock);
  ets.read(memblock,4);
  int nrColors = *reinterpret_cast<int*>(memblock);
  ets.read(memblock,4);
  int colorSpace = *reinterpret_cast<int*>(memblock);
  ets.read(memblock,4);
  _compressionType = *reinterpret_cast<int*>(memblock);
  ets.read(memblock,4);
  int compressionQuality = *reinterpret_cast<int*>(memblock);
  ets.read(memblock,4);
  _tileSizeX = *reinterpret_cast<int*>(memblock);
  ets.read(memblock,4);
  _tileSizeY = *reinterpret_cast<int*>(memblock);
  ets.read(memblock,4);
  int tileDepth = *reinterpret_cast<int*>(memblock);
  bool isRGB = nrColors > 1;

  // Read locations of tiles and file offsets
  ets.seekg(usedChunkOffset);
  for (int tile = 0; tile < nUsedChunks; ++tile) {
	  ets.seekg(4,ios::cur);
	  vector<int> curTileCoords;
   	  for (int i=0; i<nDims; i++) {
		  ets.read(memblock,4);
		  curTileCoords.push_back(*reinterpret_cast<int*>(memblock));
	  }
	  ets.read(memblockLong,8);
	  _tileOffsets.push_back(*reinterpret_cast<unsigned long long*>(memblockLong));
        ets.read(memblock,4);
	  int nrBytes = *reinterpret_cast<int*>(memblock);
	  ets.seekg(4, ios::cur);
	  _tileCoords.push_back(curTileCoords);
  }
  int maxX = 0;
  int maxY = 0;

  for (vector<vector<int> >::iterator t = _tileCoords.begin(); t != _tileCoords.end(); ++t) {
	  if ((*t)[0] > maxX) {
	  maxX = (*t)[0];
	  }
	  if ((*t)[1] > maxY) {
	  maxY = (*t)[1];
	  }
  }
  std::vector<unsigned long long> L0Dims(2,0);
  if (maxX > 1) {
	  L0Dims[0] = _tileSizeX * (maxX + 1);
  }
  else {
	  L0Dims[0] = _tileSizeX;
  }
  if (maxY > 1) {
	  L0Dims[1] = _tileSizeY * (maxY + 1);
  }
  else {
	  L0Dims[1] = _tileSizeY;
  }
  _levelDimensions.push_back(L0Dims);

  if (maxY > 1) {
	  _nrTilesY = maxY + 1;
  }
  else {
	  _nrTilesY = 1;
  }
  if (maxX > 1) {
	  _nrTilesX = maxX + 1;
  }
  else {
	  _nrTilesX = 1;
  }

  delete[] memblock;
  delete[] memblockLong;

  // Set some defaults for VSI
  _numberOfLevels = 1;
  _samplesPerPixel = 3;
  _colorType = RGB;
  _dataType = UChar;
  if (L0Dims[0]*L0Dims[1] > 0) {
    _isValid = true;
  } else {
    _isValid = false;
  }

  return L0Dims[0]*L0Dims[1];
}

char* VSIImage::decodeTile(int no, int row, int col) const {	
  int size = _tileSizeX*_tileSizeY*3;
  char* buf = new char[size];
	if (no==_tileCoords.size()) {
		std::fill(buf, buf+(3*_tileSizeX*_tileSizeY), 0);
	} else {
		ifstream ets;
		ets.open(_etsFile.c_str(), ios::in | ios::binary);
		ets.seekg(_tileOffsets[no]);
    ets.read(buf, size);
    if (_compressionType == 0) {
      return buf;
    }
    else if (_compressionType == 3) {
      JPEG2000Codec cod;
      cod.decode(buf, size, size);
    }
    else if (_compressionType == 2 || _compressionType == 5) {
      jpeg_decompress_struct cinfo;
      jpeg_error_mgr jerr; //error handling
      jpeg_source_mgr src_mem;
      jpeg_create_decompress(&cinfo);
      cinfo.err = jpeg_std_error(&jerr);      
      jpeg_mem_src(&cinfo, &src_mem, (void*)buf, size);
      jpeg_read_header(&cinfo, true);
      if (_compressionType == 2) {
        cinfo.jpeg_color_space = JCS_YCbCr;
      } else {
        cinfo.jpeg_color_space = JCS_RGB;
      }
      jpeg_start_decompress(&cinfo);
      unsigned char* outBuf = new unsigned char[size];
      unsigned char* line = outBuf;
      while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines (&cinfo, &line, 1);
        line += 3*cinfo.output_width;
      }
      jpeg_finish_decompress(&cinfo);
      jpeg_destroy_decompress(&cinfo);
      delete[] buf;
      buf = (char*)outBuf;
    }
	}
	return buf;
}

void* VSIImage::readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level) {
    if (level!=0) {
      return NULL;
    }
    unsigned char* data = new unsigned char[width*height*_samplesPerPixel];
    int tileRows = _nrTilesY;
    int tileCols = _nrTilesX;

    Box image = Box(startX, startY, width, height);
    int outputRow = 0, outputCol = 0;
    int outputRowLen = width * 3;

	  Box intersection(0,0,0,0);
    for (int row=0; row<tileRows; row++) {
      for (int col=0; col<tileCols; col++) {
        int width = _tileSizeX;
        int height = _tileSizeY;
        Box tile = Box(col * width, row * height, width, height);
        if (!tile.intersects(image)) {
          continue;
        }

        intersection = tile.intersection(image);
        int intersectionX = 0;

        if (tile.getStart()[0] < image.getStart()[0]) {
          intersectionX = image.getStart()[0] - tile.getStart()[0];
        }

		    int no=0;
		    for (vector<vector<int> >::const_iterator it = _tileCoords.begin(); it != _tileCoords.end(); ++it) {
			    if (((*it)[0] == col) && ((*it)[1] == row && ((*it)[3] == 0))) {
				    break;
			    }
			    no++;
		    }
        char* tileBuf =  decodeTile(no, row, col);
        int rowLen = 3 * (intersection.getSize()[0] < width ? intersection.getSize()[0] : width);
        int outputOffset = outputRow * outputRowLen + outputCol;
        for (int trow=0; trow<intersection.getSize()[1]; trow++) {
          int realRow = trow + intersection.getStart()[1] - tile.getStart()[1];
          int inputOffset = 3 * (realRow * width + intersectionX);
          memcpy(data+outputOffset, tileBuf+inputOffset, rowLen);
          outputOffset += outputRowLen;
        }

        outputCol += rowLen;
		    delete[] tileBuf;
      }

      if (intersection.getSize()[0] > 0 && intersection.getSize()[1] > 0) {
        outputRow += intersection.getSize()[1];
        outputCol = 0;
      }
    }
  return data;
}
