#include "DICOMImage.h"
#include <mutex>
#include <sstream>
#include "core/filetools.h"
#include "dcmtk/dcmdata/dcfilefo.h"
#include "WSIDicomInstance.h"
using namespace pathology;

DICOMImage::DICOMImage() : MultiResolutionImage() {
}

DICOMImage::~DICOMImage() {
  std::unique_lock<std::shared_mutex> l(*_openCloseMutex);
  cleanup();
  MultiResolutionImage::cleanup();
}

// We are using OpenSlides caching system instead of our own.
void DICOMImage::setCacheSize(const unsigned long long cacheSize) {

}

bool DICOMImage::initializeType(const std::string& imagePath) {
  std::unique_lock<std::shared_mutex> l(*_openCloseMutex);
  cleanup();
  std::string dirPath = core::extractFilePath(imagePath);
  std::vector<std::string> dcmFilePaths;
  core::getFiles(dirPath, "*.dcm", dcmFilePaths);
  std::vector<WSIDicomInstance*> dcmInstances;
  for (auto dcmFilePath : dcmFilePaths) {
      DcmFileFormat* dcm = new DcmFileFormat();
      OFCondition status = dcm->loadFile(OFFilename(dcmFilePath.c_str()));
      if (status.good()) {
          WSIDicomInstance* instance = new WSIDicomInstance(dcm);
          if (instance->valid()) {
              dcmInstances.push_back(instance);
          }
      }
      else {
          return _isValid;
      }
  }

  // Determine base level
  unsigned long long largestNrPixels = 0;
  WSIDicomInstance* baseLevel = nullptr;
  for (auto instance : dcmInstances) {
      std::vector<unsigned int> instanceSize = instance->getSize();
      unsigned long long nrPixels = instanceSize[0] * instanceSize[1];
      if (nrPixels > largestNrPixels) {
          baseLevel = instance;
          largestNrPixels = nrPixels;
      }
  }

  // Filter out all instances that do not have the same UIDs and order them according to the right level
  std::map<int, std::vector<WSIDicomInstance*> > downsampleToLevel;
  downsampleToLevel[1] = { baseLevel };
  for (auto instance : dcmInstances) {
      if (instance->getUID("StudyInstanceUID") == baseLevel->getUID("StudyInstanceUID") &&
          instance->getUID("SeriesInstanceUID") == baseLevel->getUID("SeriesInstanceUID") /* &&
          instance->getUID("FrameOfReferenceUID") == baseLevel->getUID("FrameOfReferenceUID")*/) {
          if (instance->getImageType() == WSIDicomInstance::DcmImageType::Volume) {
              float downsample = static_cast<float>(baseLevel->getSize()[0]) / instance->getSize()[0];
              int roundedDownsample = std::lround(downsample);
              if (std::abs(downsample - roundedDownsample) > 0.5) {
                  return _isValid;
              }
              if (downsampleToLevel.count(roundedDownsample) > 0) { // Could be a concatenation, check!
                  std::string concatUID = downsampleToLevel[roundedDownsample][0]->getUID("SOPInstanceUIDOfConcatenationSource");
                  if (!concatUID.empty()) {
                      if (concatUID == instance->getUID("SOPInstanceUIDOfConcatenationSource")) {
                          downsampleToLevel[roundedDownsample].push_back(instance);
                      }
                  }
              }
              else {
                  downsampleToLevel[roundedDownsample] = { instance };
              }
          }
          else if (instance->getImageType() == WSIDicomInstance::DcmImageType::Overview) {
              _overview = instance;
          }
          else if (instance->getImageType() == WSIDicomInstance::DcmImageType::Label) {
              _label = instance;
          }
          else {
              return _isValid;
          }
      }
  }
  for (auto it = downsampleToLevel.begin(); it != downsampleToLevel.end(); it++) {
      _levels.push_back(it->second);
      _levelDimensions.push_back({ it->second[0]->getSize()[0], it->second[0]->getSize()[1] });
  }
  _numberOfLevels = _levels.size();
  _spacing = _levels[0][0]->getPixelSpacing();
  _colorType = pathology::ColorType::RGB;
  _dataType = pathology::DataType::UChar;
  _samplesPerPixel = 3;

  _isValid = true;
  return _isValid;
}
std::string DICOMImage::getProperty(const std::string& propertyName) {
  std::string propertyValue;
  return propertyValue;
}

void* DICOMImage::readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level) {
  std::unique_lock<std::mutex> l(*_cacheMutex);
  std::vector<WSIDicomInstance*> currentLevel = _levels[level];
  long long levelW = _levelDimensions[level][0];
  long long levelH = _levelDimensions[level][1];
  unsigned short tileW = currentLevel[0]->getTileSize()[0];
  unsigned short tileH = currentLevel[0]->getTileSize()[1];
  double downsample = this->getLevelDownsample(level);

  unsigned char* temp = new unsigned char[width * height * _samplesPerPixel];
  std::fill(temp, temp + width * height * _samplesPerPixel, static_cast<unsigned char>(0));

  if (currentLevel.size() == 1) {
      long long levelStartX = std::floor(startX / downsample + 0.5);
      long long levelStartY = std::floor(startY / downsample + 0.5);
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
              unsigned char* tile = (unsigned char*)(currentLevel[0]->getFrame(ix, iy, 0, 0));

              long long ixx = (ix - levelStartX);
              long long iyy = (iy - levelStartY);
              long long lyh = levelStartY + height;
              long long lxw = levelStartX + width;
              long long ixw = ixx + tileW;
              long long rowLength = ixw > static_cast<long long>(width) ? (tileW - (ixw - width)) * _samplesPerPixel : tileW * _samplesPerPixel;
              long long tileDeltaX = 0;
              if (ixx < 0) {
                  rowLength += ixx * _samplesPerPixel;
                  tileDeltaX -= ixx * _samplesPerPixel;
                  ixx = 0;
              }
              for (unsigned int ty = 0; ty < tileH; ++ty) {
                  if ((iyy + ty >= 0) && (ixx >= 0) && (iyy + ty < static_cast<long long>(height)) && lxw > 0) {
                      long long idx = (ty + iyy) * width * _samplesPerPixel + ixx * _samplesPerPixel;
                      long long tids = (ty * tileW) * _samplesPerPixel;
                      std::copy(tile + tids + tileDeltaX, tile + tids + rowLength + tileDeltaX, temp + idx);
                  }
              }
              delete[] tile;
          }
      }

  }
  return temp;
}

void DICOMImage::cleanup() {
}