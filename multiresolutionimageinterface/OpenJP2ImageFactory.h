#ifndef _OPENJP2IMAGEFACTORY
#define _OPENJP2IMAGEFACTORY

#include "multiresolutionimageinterface_export.h"
#include "MultiResolutionImageFactory.h"

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT OpenJP2ImageFactory : public MultiResolutionImageFactory {
public:
  OpenJP2ImageFactory();

private:
  static const OpenJP2ImageFactory registerThis;
  MultiResolutionImage* readImage(const std::string& fileName) const;
  bool canReadImage(const std::string& fileName) const;
};

#endif