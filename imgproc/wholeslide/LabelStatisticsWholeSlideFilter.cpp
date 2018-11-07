#include "LabelStatisticsWholeSlideFilter.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageWriter.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "core/PathologyEnums.h"
#include "core/filetools.h"
#include <set>
#include <fstream>
#include <iostream>

LabelStatisticsWholeSlideFilter::LabelStatisticsWholeSlideFilter() :
_monitor(NULL),
_processedLevel(0),
_outPath("")
{

}

LabelStatisticsWholeSlideFilter::~LabelStatisticsWholeSlideFilter() {
}

void LabelStatisticsWholeSlideFilter::setInput(const std::shared_ptr<MultiResolutionImage>& input) {
  _input = input;
}

void LabelStatisticsWholeSlideFilter::setOutput(const std::string& outPath) {
  _outPath = outPath;
}

void LabelStatisticsWholeSlideFilter::setProcessedLevel(const unsigned int processedLevel) {
  _processedLevel = processedLevel;
}

unsigned int LabelStatisticsWholeSlideFilter::getProcessedLevel() {
  return _processedLevel;
}

void LabelStatisticsWholeSlideFilter::setProgressMonitor(ProgressMonitor* progressMonitor) {
  _monitor = progressMonitor;
}

ProgressMonitor* LabelStatisticsWholeSlideFilter::getProgressMonitor() {
  return _monitor;
}

std::vector<std::vector<float> > LabelStatisticsWholeSlideFilter::getLabelStatistics() {
  return _labelStats;
}

bool LabelStatisticsWholeSlideFilter::process() {
  _labelStats.clear();
  std::shared_ptr<MultiResolutionImage> img = _input.lock();
  std::vector<unsigned long long> dims = img->getLevelDimensions(this->_processedLevel);
  double downsample = img->getLevelDownsample(this->_processedLevel);

  std::ofstream csvfile;
  if (!_outPath.empty()) {
    csvfile.open(this->_outPath);
    if (!csvfile.is_open()) {
      std::cerr << "ERROR: Could not open file for writing" << std::endl;
      return false;
    }
  }

  unsigned int* tile = new unsigned int[512 * 512];
  for (unsigned long long t_y = 0; t_y < dims[1]; t_y += 512) {
    for (unsigned long long t_x = 0; t_x < dims[0]; t_x += 512) {
      img->getRawRegion<unsigned int>(static_cast<unsigned long long>(t_x*downsample), static_cast<unsigned long long>(t_y*downsample), 512, 512, this->_processedLevel, tile);
      for (unsigned int y = 0; y < 512; ++y) {
        for (unsigned int x = 0; x < 512; ++x) {
          unsigned int curVal = tile[y * 512 + x];
          if (curVal > 0) {
            if (curVal > _labelStats.size()) {
              _labelStats.resize(curVal, std::vector<float>(3, 0.0));
            }
            std::vector<float>& loc = _labelStats[curVal - 1];
            loc[0] += (t_x + x);
            loc[1] += (t_y + y);
            loc[2] += 1;
          }
        }
      }
    }
  }
  if (csvfile.is_open()) {
    csvfile << "Label,CoGX,CoGY,Area\n";
  }
  for (std::vector<std::vector<float> >::iterator it = _labelStats.begin(); it != _labelStats.end(); ++it) {
    float area = (*it)[2];
    if (area > 0) {
      (*it)[0] /= area;
      (*it)[1] /= area;
      if (csvfile.is_open()) {
        csvfile << static_cast<unsigned int>(it - _labelStats.begin()) + 1 << "," << (*it)[0] << "," << (*it)[1] << "," << area << std::endl;
      }
    }
  }
  if (csvfile.is_open()) {
    csvfile.close();
  }

  delete[] tile;
  return true;
}