#ifndef _TIFFIMAGEFACTORY
#define _TIFFIMAGEFACTORY

#include "multiresolutionimageinterface_export.h"
#include "MultiResolutionImageFactory.h"

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT TIFFImageFactory : public MultiResolutionImageFactory {
public:
  TIFFImageFactory();

private:
  static const TIFFImageFactory registerThis;
  MultiResolutionImage* readImage(const std::string& fileName) const;
  bool canReadImage(const std::string& fileName) const;
};

#endif