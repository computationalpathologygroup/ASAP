#include "VSIImageFactory.h"
#include "VSIImage.h"

const VSIImageFactory VSIImageFactory::registerThis;

VSIImageFactory::VSIImageFactory() : MultiResolutionImageFactory("Olympus VSI", { "vsi" }, 0) {
}

MultiResolutionImage* VSIImageFactory::readImage(const std::string& fileName) const {
  VSIImage* img = new VSIImage();
  img->initialize(fileName);
  if (img->valid()) {
    return img;
  }
  else {
    delete img;
    return NULL;
  }
}

bool VSIImageFactory::canReadImage(const std::string& fileName) const {
  VSIImage* img = new VSIImage();
  bool canOpen = img->initialize(fileName);
  delete img;
  return canOpen;
}