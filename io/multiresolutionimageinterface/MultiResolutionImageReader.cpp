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
  else if (ext == "tif" || ext == "tiff") {
    img = new TIFFImage();
  }
  else {
    img = new OpenSlideImage();
  }
  if (!img->initialize(fileName)) {
    delete img;
    img = NULL;
  }
  return img;
}