#include "ThresholdWholeSlideFilter.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageWriter.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "core/PathologyEnums.h"
#include "core/filetools.h"
#include <set>
#include <fstream>
#include <iostream>

ThresholdWholeSlideFilter::ThresholdWholeSlideFilter() :
_monitor(NULL),
_processedLevel(0),
_outPath(""),
_lowerThreshold(std::numeric_limits<float>::min()),
_upperThreshold(std::numeric_limits<float>::max()),
_component(-1)
{

}

ThresholdWholeSlideFilter::~ThresholdWholeSlideFilter() {
}

void ThresholdWholeSlideFilter::setInput(const std::shared_ptr<MultiResolutionImage>& input) {
  _input = input;
}

void ThresholdWholeSlideFilter::setOutput(const std::string& outPath) {
  _outPath = outPath;
}

void ThresholdWholeSlideFilter::setProcessedLevel(const unsigned int processedLevel) {
  _processedLevel = processedLevel;
}

unsigned int ThresholdWholeSlideFilter::getProcessedLevel() {
  return _processedLevel;
}

void ThresholdWholeSlideFilter::setProgressMonitor(ProgressMonitor* progressMonitor) {
  _monitor = progressMonitor;
}

ProgressMonitor* ThresholdWholeSlideFilter::getProgressMonitor() {
  return _monitor;
}

void ThresholdWholeSlideFilter::setLowerThreshold(const float& threshold) {
  this->_lowerThreshold = threshold;
}

void ThresholdWholeSlideFilter::setUpperThreshold(const float& threshold) {
  this->_upperThreshold = threshold;
}

float ThresholdWholeSlideFilter::getLowerThreshold() const {
  return this->_lowerThreshold;
}

float ThresholdWholeSlideFilter::getUpperThreshold() const {
  return this->_upperThreshold;
}

void ThresholdWholeSlideFilter::setComponent(const int& component) {
  this->_component = component;
}

int ThresholdWholeSlideFilter::getComponent() const {
  return this->_component;
}

bool ThresholdWholeSlideFilter::process() {
  std::shared_ptr<MultiResolutionImage> img = _input.lock();
  std::vector<unsigned long long> dims = img->getLevelDimensions(this->_processedLevel);
  double downsample = img->getLevelDownsample(this->_processedLevel);

  MultiResolutionImageWriter writer;
  unsigned int outSamplesPerPixel = img->getSamplesPerPixel();
  unsigned int inSamplesPerPixel = img->getSamplesPerPixel();
  if (_component >= inSamplesPerPixel) {
    std::cerr << "ERROR: Selected component is larger than number of input components, fallback to all components" << std::endl;
    _component = -1;
  }
  if (_component >= 0 || img->getSamplesPerPixel() == 1) {
    writer.setColorType(pathology::ColorType::Monochrome);
    outSamplesPerPixel = 1;
  } else {
    writer.setColorType(pathology::ColorType::Indexed);
    writer.setNumberOfIndexedColors(img->getSamplesPerPixel());
  }
  writer.setCompression(pathology::Compression::LZW);
  writer.setDataType(pathology::DataType::UChar);
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

  float* tile = new float[512 * 512 * img->getSamplesPerPixel()];
  unsigned char* out_tile = new unsigned char[512 * 512 * outSamplesPerPixel];
  for (unsigned long long t_y = 0; t_y < dims[1]; t_y += 512) {
    for (unsigned long long t_x = 0; t_x < dims[0]; t_x += 512) {
      img->getRawRegion<float>(static_cast<unsigned long long>(t_x*downsample), static_cast<unsigned long long>(t_y*downsample), 512, 512, this->_processedLevel, tile);
      for (unsigned int y = 0; y < 512; ++y) {
        for (unsigned int x = 0; x < 512; ++x) {
          for (unsigned int c = 0; c < img->getSamplesPerPixel(); ++c) {
            if (c == _component) {
              float curVal = tile[y * 512 * inSamplesPerPixel + x * inSamplesPerPixel + c];
              if (curVal >= _lowerThreshold && curVal < _upperThreshold) {
                out_tile[y * 512 + x] = 1;
              }
              else {
                out_tile[y * 512 + x] = 0;
              }
            }
            else if (_component < 0) {
              float curVal = tile[y * 512 * inSamplesPerPixel + x * inSamplesPerPixel + c];
              if (curVal >= _lowerThreshold && curVal < _upperThreshold) {
                out_tile[y * 512 * inSamplesPerPixel + x * inSamplesPerPixel + c] = 1;
              }
              else {
                out_tile[y * 512 * inSamplesPerPixel + x * inSamplesPerPixel + c] = 0;
              }
            }
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