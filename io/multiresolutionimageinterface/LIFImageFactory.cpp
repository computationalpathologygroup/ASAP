#include "LIFImageFactory.h"
#include "LIFImage.h"

const LIFImageFactory LIFImageFactory::registerThis;

LIFImageFactory::LIFImageFactory() : MultiResolutionImageFactory("lif") {
}

MultiResolutionImage* LIFImageFactory::readImage(const std::string& fileName) const {
  LIFImage* img = new LIFImage();
  img->initialize(fileName);
  if (img->valid()) {
    return img;
  }
  else {
    delete img;
    return NULL;
  }
}

bool LIFImageFactory::canReadImage(const std::string& fileName) const {
  LIFImage* img = new LIFImage();
  bool canOpen = img->initialize(fileName);
  delete img;
  return canOpen;
}