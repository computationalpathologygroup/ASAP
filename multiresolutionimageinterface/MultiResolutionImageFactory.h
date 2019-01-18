#ifndef _MultiResolutionImageFactory
#define _MultiResolutionImageFactory

#include <string>
#include <map>
#include <vector>
#include <set>
#include "multiresolutionimageinterface_export.h"

class MultiResolutionImage;

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT MultiResolutionImageFactory {

public:
  MultiResolutionImageFactory(const std::string& factoryName, const std::set<std::string>& supported_extensions, const unsigned int priority);
  static MultiResolutionImage* openImage(const std::string& fileName, const std::string factoryName = std::string("default"));
  static void registerExternalFileFormats();
  static std::vector<std::pair<std::string, std::set<std::string> > > getLoadedFactoriesAndSupportedExtensions();
  static std::set<std::string> getAllSupportedExtensions();
  bool operator< (const MultiResolutionImageFactory &other) const;

protected:
  typedef std::map<std::string, std::pair<std::set<std::string>, MultiResolutionImageFactory*> > FactoryMap;
  static FactoryMap& registry();
  static void addSupportedExtensions(const std::set<std::string>& extensions);
  static MultiResolutionImage* openImageWithFactory(const std::string& fileName, const MultiResolutionImageFactory* factory);
  virtual MultiResolutionImage* readImage(const std::string& fileName) const = 0;
  virtual bool canReadImage(const std::string& fileName) const = 0;

private:
  static bool _externalFormatsRegistered;
  static std::set<std::string> _allSupportedExtensions;
  const std::string _factoryName;
  const unsigned int _priority;
};

#endif