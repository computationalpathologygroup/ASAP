#ifndef _OPENSLIDEIMAGEFACTORY
#define _OPENSLIDEIMAGEFACTORY

#include "openslidefileformat_export.h"
#include "MultiResolutionImageFactory.h"

class OPENSLIDEFILEFORMAT_EXPORT OpenSlideImageFactory : public MultiResolutionImageFactory {
public:
  OpenSlideImageFactory();

private:
  MultiResolutionImage* readImage(const std::string& fileName) const;
  bool canReadImage(const std::string& fileName) const;
};

extern "C" {
  OPENSLIDEFILEFORMAT_EXPORT void filetypeLoad();
}

#endif