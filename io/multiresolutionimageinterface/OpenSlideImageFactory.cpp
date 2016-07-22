#include "OpenSlideImageFactory.h"
#include "OpenSlideImage.h"

OpenSlideImageFactory::OpenSlideImageFactory() : MultiResolutionImageFactory("svs;tif;tiff;mrxs;vms;vmu;ndpi;scn;svslide;bif") {
}

MultiResolutionImage* OpenSlideImageFactory::readImage(const std::string& fileName) const {
  OpenSlideImage* img = new OpenSlideImage();
  img->initialize(fileName);
  if (img->valid()) {
    return img;
  }
  else {
    delete img;
    return NULL;
  }
}

bool OpenSlideImageFactory::canReadImage(const std::string& fileName) const {
  OpenSlideImage* img = new OpenSlideImage();
  bool canOpen = img->initialize(fileName);
  delete img;
  return canOpen;
}

void filetypeLoad()
{
  static OpenSlideImageFactory filetypeFactory;
}