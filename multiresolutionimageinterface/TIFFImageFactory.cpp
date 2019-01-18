#include "TIFFImageFactory.h"
#include "TIFFImage.h"
#include "tiffio.h"

const TIFFImageFactory TIFFImageFactory::registerThis;

TIFFImageFactory::TIFFImageFactory() : MultiResolutionImageFactory("ASAP TIF", {"tif", "tiff"}, 0) {
  TIFFSetWarningHandler(NULL);
  TIFFSetErrorHandler(NULL);
}

MultiResolutionImage* TIFFImageFactory::readImage(const std::string& fileName) const {
  TIFFImage* img = new TIFFImage();
  img->initialize(fileName);
  if (img->valid()) {
    return img;
  }
  else {
    delete img;
    return NULL;
  }
}

bool TIFFImageFactory::canReadImage(const std::string& fileName) const {
  TIFFImage* img = new TIFFImage();
  bool canOpen = img->initialize(fileName);
  delete img;
  return canOpen;
}