#include "OpenSlideImageFactory.h"
#include "OpenSlideImage.h"

OpenSlideImageFactory::OpenSlideImageFactory() : MultiResolutionImageFactory("OpenSlide Formats", { "svs", "tif", "tiff", "mrxs", "vms", "vmu", "ndpi", "scn", "svslide", "bif" }, 1) {
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