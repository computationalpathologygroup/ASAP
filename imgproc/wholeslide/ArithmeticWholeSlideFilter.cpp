#include "ArithmeticWholeSlideFilter.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageWriter.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "core/PathologyEnums.h"
#include "core/filetools.h"
#include "core/stringconversion.h"
#include <set>
#include <fstream>
#include <iostream>

ArithmeticWholeSlideFilter::ArithmeticWholeSlideFilter() :
_monitor(NULL),
_processedLevel(0),
_outPath(""),
_expression("")
{

}

ArithmeticWholeSlideFilter::~ArithmeticWholeSlideFilter() {
}

void ArithmeticWholeSlideFilter::setInput(const std::shared_ptr<MultiResolutionImage>& input) {
  _input = input;
}

void ArithmeticWholeSlideFilter::setOutput(const std::string& outPath) {
  _outPath = outPath;
}

void ArithmeticWholeSlideFilter::setProcessedLevel(const unsigned int processedLevel) {
  _processedLevel = processedLevel;
}

unsigned int ArithmeticWholeSlideFilter::getProcessedLevel() {
  return _processedLevel;
}

void ArithmeticWholeSlideFilter::setProgressMonitor(ProgressMonitor* progressMonitor) {
  _monitor = progressMonitor;
}

ProgressMonitor* ArithmeticWholeSlideFilter::getProgressMonitor() {
  return _monitor;
}

void ArithmeticWholeSlideFilter::setExpression(const std::string& expression) {
  this->_expression = expression;
}

std::string ArithmeticWholeSlideFilter::getExpression() const {
  return this->_expression;
}

bool ArithmeticWholeSlideFilter::process() {
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
  if (writer.openFile(_outPath) != 0) {
    std::cerr << "ERROR: Could not open file for writing" << std::endl;
    return false;
  }
  writer.setProgressMonitor(_monitor);
  writer.writeImageInformation(dims[0], dims[1]);

  std::vector<unsigned char> labels;
  std::vector<std::string> stringLabels;
  core::split(_expression, stringLabels, ",");
  labels.resize(core::fromstring<unsigned int>(stringLabels.back()), 0);
  for (unsigned int i = 0; i < stringLabels.size() - 1; ++i) {
    labels[core::fromstring<unsigned int>(stringLabels[i])] = 1;
  }
  unsigned int* tile = new unsigned int[512 * 512];
  unsigned int* out_tile = new unsigned int[512 * 512];
  for (unsigned long long t_y = 0; t_y < dims[1]; t_y += 512) {
    for (unsigned long long t_x = 0; t_x < dims[0]; t_x += 512) {
      img->getRawRegion<unsigned int>(static_cast<unsigned long long>(t_x*downsample), static_cast<unsigned long long>(t_y*downsample), 512, 512, this->_processedLevel, tile);
      for (unsigned int y = 0; y < 512; ++y) {
        for (unsigned int x = 0; x < 512; ++x) {
          float curVal = tile[y * 512 + x];
          if (curVal > 0 && labels[curVal]==0) {
            out_tile[y * 512 + x] = curVal;
          }
          else {
            out_tile[y * 512 + x] = 0;
          }
        }
      }
      writer.writeBaseImagePart(reinterpret_cast<void*>(out_tile));
    }
  }
  writer.finishImage();

  delete[] tile;
  delete[] out_tile;
  return true;
}