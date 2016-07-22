#ifndef _LIFIMAGEFACTORY
#define _LIFIMAGEFACTORY

#include "multiresolutionimageinterface_export.h"
#include "MultiResolutionImageFactory.h"

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT LIFImageFactory : public MultiResolutionImageFactory {
public:
  LIFImageFactory();

private:
  static const LIFImageFactory registerThis;
  MultiResolutionImage* readImage(const std::string& fileName) const;
  bool canReadImage(const std::string& fileName) const;
};

#endif