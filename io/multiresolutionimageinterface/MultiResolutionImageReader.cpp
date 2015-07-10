#include "MultiResolutionImageReader.h"
#include "MultiResolutionImage.h"
#include "core/filetools.h"

#include "VSIImage.h"
#include "TIFFImage.h"
#include "LIFImage.h"
#include "OpenSlideImage.h"

using std::string;

MultiResolutionImageReader::MultiResolutionImageReader()
{
}

MultiResolutionImageReader::~MultiResolutionImageReader() {
}

MultiResolutionImage* MultiResolutionImageReader::open(const std::string& fileName) {
  std::string ext = core::extractFileExtension(fileName);
  MultiResolutionImage* img = NULL;
  if (ext == "vsi") {
    img = new VSIImage();
  }
  else if (ext == "lif") {
    img = new LIFImage();
  }
  else {
    img = new OpenSlideImage();
  }
  if (!img->initialize(fileName) || (dynamic_cast<OpenSlideImage*>(img) && img->getNumberOfLevels() < 2)) {
    delete img;
    img = NULL;
    if (ext == "tif") {
      img = new TIFFImage();      
      if (!img->initialize(fileName)) {
        delete img;
        img = NULL;
      }
    }
  }
  return img;
}