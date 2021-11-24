#include "DICOMImage.h"
#include <shared_mutex>
#include <sstream>
#include "core/filetools.h"

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dcfilefo.h"
#include "dcmtk/ofstd/ofcond.h"
#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcuid.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmdata/dcxfer.h"
#include "dcmtk/dcmdata/dcmetinf.h"
#include "dcmtk/dcmdata/dcrlerp.h"
#include "dcmtk/dcmdata/dcrledrg.h"

// JPEG en/decoding
#include "dcmtk/dcmjpeg/djdecode.h"
#include "dcmtk/dcmjpeg/djencode.h"

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
      DcmFileFormat dcm;
      OFCondition status = dcm.loadFile(OFFilename(dcmFilePath.c_str()));
      if (status.good()) {
          OFString msSOPClassUID;
          std::vector<std::string> supportedTransferSyntax = { UID_JPEGProcess1TransferSyntax, UID_JPEGProcess2_4TransferSyntax, UID_JPEG2000LosslessOnlyTransferSyntax, UID_JPEG2000TransferSyntax };
          std::vector<DcmTagKey> requiredTags = { DCM_StudyInstanceUID, DCM_SeriesInstanceUID, DCM_FrameOfReferenceUID, DCM_Rows, DCM_Columns,
                                                  DCM_SamplesPerPixel, DCM_PhotometricInterpretation, DCM_ImageType, DCM_TotalPixelMatrixColumns,
                                                  DCM_TotalPixelMatrixRows, DCM_NumberOfFrames,  DCM_SharedFunctionalGroupsSequence, DCM_OpticalPathSequence};
          DcmMetaInfo* metainfo = dcm.getMetaInfo();
          DcmDataset* dcmDataset = dcm.getDataset();
          metainfo->findAndGetOFString(DCM_MediaStorageSOPClassUID, msSOPClassUID);
          if (msSOPClassUID == UID_VLWholeSlideMicroscopyImageStorage) {
              dcmFiles.push_back(dcm);
              DcmXfer transferSyntax = dcmDataset->getOriginalXfer();
              if (std::find(supportedTransferSyntax.begin(), supportedTransferSyntax.end(), transferSyntax.getXferID()) == supportedTransferSyntax.end()) {
                  return _isValid;
              }
              OFString tagValue;
              for (auto tag : requiredTags) {
                  if (dcmDataset->findAndGetOFString(tag, tagValue).bad()) {
                      return _isValid;
                  }
                  OFString imageType;
                  dcmDataset->findAndGetOFString(DCM_ImageType, imageType, 2);
                  if (imageType == "VOLUME") {
                      DicomImage* image = new DicomImage(&dcm, EXS_Unknown, CIF_UsePartialAccessToPixelData);
                  }
              }
          }
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