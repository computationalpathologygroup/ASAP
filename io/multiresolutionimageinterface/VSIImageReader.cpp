#include "VSIImageReader.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include "core/filetools.h"
#include <string.h>

// Include DCMTK LIBJPEG for lossy and lossless JPEG compression
extern "C" {
#define boolean ijg_boolean
#include "dcmjpeg/libijg8/jpeglib8.h"
#include "jpeg_mem_src.h"
#undef boolean
}

#include "JPEG2000Codec.h"

using namespace std;

VSIImageReader::VSIImageReader() : 
	_vsiFileName(""), _etsFile(""), _tileOffsets(), _tileCoords(),
	_tileSizeX(0), _tileSizeY(0), _imageSizeX(0), _imageSizeY(0), _nrTilesX(0),
	_nrTilesY(0), _pixelSizeX(0),_compressionType(0),_pixelSizeY(0)
{
}

VSIImageReader::~VSIImageReader() {
}

void VSIImageReader::clean() {
	_vsiFileName = "";
	_etsFile = "";
	_tileCoords.clear();
	_tileOffsets.clear();
	_tileSizeX = 0;
	_tileSizeY = 0;
	_imageSizeX = 0;
	_imageSizeY = 0;
	_nrTilesX = 0;
	_nrTilesY = 0;
	_pixelSizeX = 0;
	_pixelSizeY = 0;
  _compressionType = 0;
}

void VSIImageReader::open(const std::string& filePath) { 
	clean();
	if (!core::fileExists(filePath)) {
		return;
	}
	_vsiFileName = filePath;
  string pth = core::extractFilePath(filePath);
	string fileName = core::extractFileName(filePath);	
  string baseName = core::extractBaseName(filePath);    
	vector<string> etsFiles;
	core::getFiles(core::completePath("_" + baseName + "_",pth),"*.ets",etsFiles, true);
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
	  _imageSizeX = 0;
	  _imageSizeY = 0;
	  _nrTilesX = 0;
	  _nrTilesY = 0;
	  _pixelSizeX = 0;
	  _pixelSizeY = 0;
    _compressionType = 0;
	}
	ifstream ets;
	ets.open(_etsFile.c_str(), ios::in | ios::binary);
  parseETSFile(ets);
  ets.close();	
}

unsigned long long VSIImageReader::parseETSFile(std::ifstream& ets) {
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

  if (maxX > 1) {
	  _imageSizeX = _tileSizeX * (maxX + 1);
  }
  else {
	  _imageSizeX = _tileSizeX;
  }
  if (maxY > 1) {
	  _imageSizeY = _tileSizeY * (maxY + 1);
  }
  else {
	  _imageSizeY = _tileSizeY;
  }

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
  return _imageSizeX*_imageSizeY;
};


void VSIImageReader::close() {
	clean();
}

const bool VSIImageReader::isOpen() const {
	return _tileOffsets.size() > 0;
}

const std::vector<unsigned long long> VSIImageReader::getDimensions() const {
  std::vector<unsigned long long> dims;
  dims.push_back(_imageSizeX);
  dims.push_back(_imageSizeY);
  return dims;
}

VSIImageReader::Region::Region(unsigned int x, unsigned int y, unsigned int w, unsigned int h) :
	x(x), y(y), w(w), h(h) 
{}

bool VSIImageReader::Region::intersects(const VSIImageReader::Region &r) const {
    unsigned int  tw = this->w;
    unsigned int  th = this->h;
    unsigned int  rw = r.w;
    unsigned int  rh = r.h;
    if (rw <= 0 || rh <= 0 || tw <= 0 || th <= 0) {
      return false;
    }
    unsigned int tx = this->x;
    unsigned int ty = this->y;
    unsigned int rx = r.x;
    unsigned int ry = r.y;
    rw += rx;
    rh += ry;
    tw += tx;
    th += ty;
    bool rtn = ((rw < rx || rw > tx) && (rh < ry || rh > ty) &&
      (tw < tx || tw > rx) && (th < ty || th > ry));
    return rtn;
  }

  VSIImageReader::Region VSIImageReader::Region::intersection(const VSIImageReader::Region& r) const {
    int x = max(this->x, r.x);
    int y = max(this->y, r.y);
    int w = min(this->x + this->w, r.x + r.w) - x;
    int h = min(this->y + this->h, r.y + r.h) - y;

    if (w < 0) w = 0;
    if (h < 0) h = 0;

    return Region(x, y, w, h);
  }

char* VSIImageReader::decodeTile(int no, int row, int col) const {	
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
      cod.decode(buf,size);
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

void VSIImageReader::readRegion(const unsigned long long& startX, const unsigned long long& startY, const unsigned long long& width, 
								const unsigned long long& height, unsigned char* data) const {

      int tileRows = _nrTilesY;
      int tileCols = _nrTilesX;

      VSIImageReader::Region image = Region(startX, startY, width, height);
      int outputRow = 0, outputCol = 0;
      int outputRowLen = width * 3;

	  Region intersection(0,0,0,0);
      for (int row=0; row<tileRows; row++) {
        for (int col=0; col<tileCols; col++) {
          int width = _tileSizeX;
          int height = _tileSizeY;
          Region tile = Region(col * width, row * height, width, height);
          if (!tile.intersects(image)) {
            continue;
          }

          intersection = tile.intersection(image);
          int intersectionX = 0;

          if (tile.x < image.x) {
            intersectionX = image.x - tile.x;
          }

		      int no=0;
		      for (vector<vector<int> >::const_iterator it = _tileCoords.begin(); it != _tileCoords.end(); ++it) {
			      if (((*it)[0] == col) && ((*it)[1] == row && ((*it)[3] == 0))) {
				      break;
			      }
			      no++;
		      }
          char* tileBuf =  decodeTile(no, row, col);
          int rowLen = 3 * min((int)intersection.w, width);

          int outputOffset = outputRow * outputRowLen + outputCol;
          for (int trow=0; trow<intersection.h; trow++) {
            int realRow = trow + intersection.y - tile.y;
            int inputOffset = 3 * (realRow * width + intersectionX);
            memcpy(data+outputOffset, tileBuf+inputOffset, rowLen);
            outputOffset += outputRowLen;
          }

          outputCol += rowLen;
		      delete[] tileBuf;
        }

        if (intersection.w > 0 && intersection.h > 0) {
          outputRow += intersection.h;
          outputCol = 0;
        }
      }
}