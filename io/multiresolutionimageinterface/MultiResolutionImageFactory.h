#ifndef _MultiResolutionImageFactory
#define _MultiResolutionImageFactory

#include <string>
#include <map>
#include "multiresolutionimageinterface_export.h"

class MultiResolutionImage;

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT MultiResolutionImageFactory {

public:
  MultiResolutionImageFactory(const std::string& supported_extensions);
  static MultiResolutionImage* openImage(const std::string& fileName);
  static void registerExternalFileFormats();

protected:
  typedef std::map<std::string, MultiResolutionImageFactory*> FactoryMap;
  static FactoryMap& registry();
  virtual MultiResolutionImage* readImage(const std::string& fileName) const = 0;
  virtual bool canReadImage(const std::string& fileName) const = 0;

private:
  static bool _externalFormatsRegistered;
};

#endif