//---------------------------------------------------------------------------
#ifndef _MultiResolutionImageReader
#define _MultiResolutionImageReader
#include "config/pathology_config.h"
#include <string>

class MultiResolutionImage;

class EXPORT_MULTIRESOLUTIONIMAGEINTERFACE MultiResolutionImageReader {

public:
  MultiResolutionImageReader();
  ~MultiResolutionImageReader();

  //! Opens the slide file and keeps a reference to it
  MultiResolutionImage* open(const std::string& fileName);

};

#endif