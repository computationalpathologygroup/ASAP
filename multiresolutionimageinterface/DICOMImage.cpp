#include "DICOMImage.h"
#include <shared_mutex>
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
  std::vector<DcmFileFormat> dcmFiles;
  for (auto dcmFilePath : dcmFilePaths) {
      DcmFileFormat* dcm = new DcmFileFormat();
      OFCondition status = dcm->loadFile(OFFilename(dcmFilePath.c_str()));
      if (status.good()) {
          WSIDicomInstance instance;
          instance.initialize(dcm);
      }
      else {
          return _isValid;
      }
  }
  return _isValid;
}
std::string DICOMImage::getProperty(const std::string& propertyName) {
  std::string propertyValue;
  return propertyValue;
}

void* DICOMImage::readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level) {
  return nullptr;
}

void DICOMImage::cleanup() {
}