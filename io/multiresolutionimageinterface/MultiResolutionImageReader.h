//---------------------------------------------------------------------------
#ifndef _MultiResolutionImageReader
#define _MultiResolutionImageReader
#include "multiresolutionimageinterface_export.h"
#include <string>

class MultiResolutionImage;

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT MultiResolutionImageReader {

public:
  MultiResolutionImageReader();
  ~MultiResolutionImageReader();

  //! Opens the slide file and keeps a reference to it
  MultiResolutionImage* open(const std::string& fileName, const std::string factoryName = std::string("default"));

};

#endif