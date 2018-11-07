#include "DistanceTransformWholeSlideFilter.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageWriter.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "core/PathologyEnums.h"
#include "core/filetools.h"
#include <set>
#include <iostream>
#include <cmath>

DistanceTransformWholeSlideFilter::DistanceTransformWholeSlideFilter() :
_monitor(NULL),
_processedLevel(0),
_outPath("")
{

}

DistanceTransformWholeSlideFilter::~DistanceTransformWholeSlideFilter() {
}

void DistanceTransformWholeSlideFilter::setInput(const std::shared_ptr<MultiResolutionImage>& input) {
  _input = input;
}

void DistanceTransformWholeSlideFilter::setOutput(const std::string& outPath) {
  _outPath = outPath;
}

void DistanceTransformWholeSlideFilter::setProcessedLevel(const unsigned int processedLevel) {
  _processedLevel = processedLevel;
}

unsigned int DistanceTransformWholeSlideFilter::getProcessedLevel() {
  return _processedLevel;
}

void DistanceTransformWholeSlideFilter::setProgressMonitor(ProgressMonitor* progressMonitor) {
  _monitor = progressMonitor;
}

ProgressMonitor* DistanceTransformWholeSlideFilter::getProgressMonitor() {
  return _monitor;
}

bool DistanceTransformWholeSlideFilter::process() const {
  std::shared_ptr<MultiResolutionImage> img = _input.lock();
  std::vector<unsigned long long> dims = img->getLevelDimensions(this->_processedLevel);
  double downsample = img->getLevelDownsample(this->_processedLevel);

  MultiResolutionImageWriter writer;
  writer.setColorType(pathology::ColorType::Monochrome);
  writer.setCompression(pathology::Compression::LZW);
  writer.setDataType(pathology::DataType::UInt32);
  writer.setInterpolation(pathology::Interpolation::NearestNeighbor);
  writer.setTileSize(512);
  std::vector<double> spacing = img->getSpacing();
  if (!spacing.empty()) {
    spacing[0] *= downsample;
    spacing[1] *= downsample;
    writer.setSpacing(spacing);
  }
  writer.setProgressMonitor(_monitor);
  std::string firstPassFile = _outPath;
  std::string basename = core::extractBaseName(_outPath);
  core::changeBaseName(firstPassFile, basename + "_firstpass");

  unsigned int* buffer_t_x = new unsigned int[512];
  unsigned int* buffer_t_y = new unsigned int[512 * dims[0]];
  unsigned char* tile = new unsigned char[512 * 512];
  unsigned int* out_tile = new unsigned int[512 * 512];
  unsigned int maxDist = (dims[0] + dims[1] / 2) + 1;
  std::fill(out_tile, out_tile + 512 * 512, maxDist);
  std::fill(buffer_t_x, buffer_t_x + 512, maxDist);
  std::fill(buffer_t_y, buffer_t_y + 512 * dims[0], maxDist);

  if (writer.openFile(firstPassFile) != 0) {
    std::cerr << "ERROR: Could not open file for writing" << std::endl;
    return false;
  }
  writer.writeImageInformation(dims[0], dims[1]);

  // Forward pass
  for (unsigned long long t_y = 0; t_y < dims[1]; t_y += 512) {
    std::fill(buffer_t_x, buffer_t_x + 512, maxDist);
    for (unsigned long long t_x = 0; t_x < dims[0]; t_x += 512) {
      img->getRawRegion<unsigned char>(static_cast<unsigned long long>(t_x*downsample), static_cast<unsigned long long>(t_y*downsample), 512, 512, this->_processedLevel, tile);
      std::fill(out_tile, out_tile + 512 * 512, (dims[0] + dims[1] / 2) + 1);
      int startX = 0;
      int startY = 0;
      if (t_x == 0) {
        startX = 1;
      }
      if (t_y == 0) {
        startY = 1;
      }
      for (int y = startY; y < 512; ++y) {
        for (int x = startX; x < 512; ++x) {
          unsigned int curPos = y * 512 + x;
          unsigned char curVal = tile[curPos];
          if (curVal == 1) {
            out_tile[curPos] = 0;
          }
          else {
            unsigned int upVal; 
            if (y == 0) {
              upVal = buffer_t_y[t_x + x];
            }
            else {
              upVal = out_tile[curPos - 512];
            }
            unsigned int leftVal;
            if (x == 0) {
              leftVal = buffer_t_x[y];
            }
            else {
              leftVal = out_tile[curPos - 1];
            }
            unsigned int newDist = std::min(upVal, leftVal) + 1;
            if (newDist < maxDist) {
              out_tile[curPos] = newDist;
            }
          }
          if (x == 511) {
            buffer_t_x[y] = out_tile[y * 512 + x];
          }
          if (y == 511) {
            buffer_t_y[t_x + x] = out_tile[y * 512 + x];
          }
        }
      }
      writer.writeBaseImagePart(reinterpret_cast<void*>(out_tile));
    }
  }
  std::fill(buffer_t_y, buffer_t_y + 512 * dims[0], maxDist);
  writer.finishImage();

  // Backward pass
  MultiResolutionImageReader reader = MultiResolutionImageReader();
  MultiResolutionImage* firstPass = reader.open(firstPassFile);
  if (writer.openFile(_outPath) != 0) {
    std::cerr << "ERROR: Could not open file for writing" << std::endl;
    return false;
  }
  writer.setProgressMonitor(_monitor);
  writer.writeImageInformation(dims[0], dims[1]);
  long long start_t_y = std::ceil(dims[1] / 512.)*512 - 512;
  long long start_t_x = std::ceil(dims[0] / 512.)*512 - 512;
  for (long long t_y = start_t_y; t_y >= 0; t_y -= 512) {
    std::fill(buffer_t_x, buffer_t_x + 512, maxDist);
    for (long long t_x = start_t_x; t_x >= 0; t_x -= 512) {
      firstPass->getRawRegion<unsigned int>(static_cast<unsigned long long>(t_x*downsample), static_cast<unsigned long long>(t_y*downsample), 512, 512, 0, out_tile);
      int startX = 511;
      int startY = 511;
      for (int y = 511; y >= 0; --y) {
        for (int x = 511; x >= 0; --x) {
          unsigned int curPos = y * 512 + x;
          unsigned int downVal;
          if (t_y + y + 1 >= dims[1]) {
            downVal = maxDist;
          }
          else if (y == 511) {
            downVal = buffer_t_y[t_x + x];
          }
          else {
            downVal = out_tile[curPos + 512];
          }
          unsigned int rightVal;
          if (t_x + x + 1 >= dims[0]) {
            rightVal = maxDist;
          }
          else if (x == 511) {
            rightVal = buffer_t_x[y];
          }
          else {
            rightVal = out_tile[curPos + 1];
          }
          unsigned int newDist = std::min(downVal, rightVal) + 1;
          if (newDist < out_tile[curPos]) {
            out_tile[curPos] = newDist;
          }
          if (x == 0) {
            buffer_t_x[y] = out_tile[y * 512 + x];
          }
          if (y == 0) {
            buffer_t_y[t_x + x] = out_tile[y * 512 + x];
          }
        }
      }
      writer.writeBaseImagePartToLocation(reinterpret_cast<void*>(out_tile), t_x, t_y);
    }
  }
  std::fill(buffer_t_y, buffer_t_y + 512 * dims[0], maxDist);
  writer.finishImage();

  delete firstPass;
  delete[] buffer_t_x;
  delete[] buffer_t_y;
  delete[] tile;
  delete[] out_tile;
  core::deleteFile(firstPassFile);
  return true;
}