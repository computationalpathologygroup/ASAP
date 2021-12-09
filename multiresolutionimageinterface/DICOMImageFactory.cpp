#include "DICOMImageFactory.h"
#include "DICOMImage.h"
#include "dcmtk/dcmjpeg/djdecode.h"

DICOMImageFactory::DICOMImageFactory() : MultiResolutionImageFactory("DICOM Formats", { "dcm" }, 1) {
    DJDecoderRegistration::registerCodecs();
}

DICOMImageFactory::~DICOMImageFactory()
{
    DJDecoderRegistration::cleanup();
}

MultiResolutionImage* DICOMImageFactory::readImage(const std::string& fileName) const {
  DICOMImage* img = new DICOMImage();
  img->initialize(fileName);
  if (img->valid()) {
    return img;
  }
  else {
    delete img;
    return NULL;
  }
}

bool DICOMImageFactory::canReadImage(const std::string& fileName) const {
  DICOMImage* img = new DICOMImage();
  bool canOpen = img->initialize(fileName);
  delete img;
  return canOpen;
}

void filetypeLoad()
{
  static DICOMImageFactory filetypeFactory;
}