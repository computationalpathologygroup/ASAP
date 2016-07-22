#ifndef _VSIIMAGEFACTORY
#define _VSIIMAGEFACTORY

#include "multiresolutionimageinterface_export.h"
#include "MultiResolutionImageFactory.h"

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT VSIImageFactory : public MultiResolutionImageFactory {
public:
  VSIImageFactory();

private:
  static const VSIImageFactory registerThis;
  MultiResolutionImage* readImage(const std::string& fileName) const;
  bool canReadImage(const std::string& fileName) const;
};

#endif