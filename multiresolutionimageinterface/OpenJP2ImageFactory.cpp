#include "OpenJP2ImageFactory.h"
#include "OpenJP2Image.h"

const OpenJP2ImageFactory OpenJP2ImageFactory::registerThis;

OpenJP2ImageFactory::OpenJP2ImageFactory() : MultiResolutionImageFactory("ASAP JP2", { "jp2"}, 1) {
}

MultiResolutionImage* OpenJP2ImageFactory::readImage(const std::string& fileName) const {
  OpenJP2Image * img = new OpenJP2Image();
  img->initialize(fileName);
  if (img->valid()) {
    return img;
  }
  else {
    delete img;
    return NULL;
  }
}

bool OpenJP2ImageFactory::canReadImage(const std::string& fileName) const {
  OpenJP2Image* img = new OpenJP2Image();
  bool canOpen = img->initialize(fileName);
  delete img;
  return canOpen;
}