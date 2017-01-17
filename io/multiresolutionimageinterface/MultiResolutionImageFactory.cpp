#include "MultiResolutionImageFactory.h"
#include "MultiResolutionImage.h"
#include "VSIImage.h"
#include "core/filetools.h"
#include "core/stringconversion.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

bool MultiResolutionImageFactory::_externalFormatsRegistered = false;

MultiResolutionImageFactory::FactoryMap& MultiResolutionImageFactory::registry() {
  static FactoryMap typeRegistry;
  return typeRegistry;
}

MultiResolutionImageFactory::MultiResolutionImageFactory(const std::string& supported_extensions) {
  registry().insert(std::make_pair(supported_extensions, this));
}

MultiResolutionImage* MultiResolutionImageFactory::openImage(const std::string& fileName) {
  MultiResolutionImageFactory::registerExternalFileFormats();
  std::string extension = core::extractFileExtension(fileName);
  for (std::map<std::string, MultiResolutionImageFactory*>::const_iterator it = registry().begin(); it != registry().end(); ++it) {
    std::vector<std::string> factoryExtensions;
    core::split(it->first, factoryExtensions, ";");
    if (std::find(factoryExtensions.begin(), factoryExtensions.end(), extension) != factoryExtensions.end()) {
      MultiResolutionImage* img = it->second->readImage(fileName);
      if (img) {
        if ((img->getNumberOfLevels() > 1 || dynamic_cast<VSIImage*>(img) != NULL) || (img->getNumberOfLevels() == 1 && img->getLevelDimensions(0)[0] < 4096)) {
          return img;
        }
        else {
          delete img;
        }
      }
    }
  }
  return NULL;
}

typedef void (*FileFormatLoader)();

void MultiResolutionImageFactory::registerExternalFileFormats() {
  if (MultiResolutionImageFactory::_externalFormatsRegistered) {
    return;
  }
  std::string pathStr;
#ifdef _WIN32
  char path[2048];
  HMODULE hm = NULL;

  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    (LPCSTR)&MultiResolutionImageFactory::registerExternalFileFormats,
    &hm))
  {
    int ret = GetLastError();
    fprintf(stderr, "GetModuleHandle returned %d\n", ret);
  }
  GetModuleFileNameA(hm, path, sizeof(path));
  pathStr = std::string(path);
#else
  Dl_info dlInfo;
  dladdr((void*)&MultiResolutionImageFactory::registerExternalFileFormats, &dlInfo);
  pathStr = std::string(dlInfo.dli_fname);
#endif
  std::string rootDir = core::extractFilePath(pathStr);
  std::vector<std::string> formatPlugins;
#ifdef _WIN32
  std::string fileFormatPluginDir = core::completePath("formats", rootDir);  
  core::getFiles(fileFormatPluginDir, "*.dll", formatPlugins);
#else
  std::string fileFormatPluginDir = core::completePath("bin/formats", core::upOneLevel(rootDir));  
  core::getFiles(fileFormatPluginDir, "*.so", formatPlugins);
#endif
  for (std::vector<std::string>::const_iterator it = formatPlugins.begin(); it != formatPlugins.end(); ++it) {
#ifdef _WIN32
    SetDllDirectory(rootDir.c_str());
    HMODULE formatLib = LoadLibrary(it->c_str());
    if (formatLib) {
      FileFormatLoader loadfunction = (FileFormatLoader)GetProcAddress(formatLib, "filetypeLoad");
      if (loadfunction) {
        (*loadfunction)();
      }
    }
    else {
      DWORD test = GetLastError();
    }
#else
    void *hndl = dlopen(core::completePath((*it), fileFormatPluginDir).c_str(), RTLD_NOW);
    if(hndl != NULL){
      FileFormatLoader loadfunction = (FileFormatLoader)dlsym(hndl, "filetypeLoad");
      if (loadfunction) {
        (*loadfunction)();
      }
    }
#endif
  }
  MultiResolutionImageFactory::_externalFormatsRegistered = true;
}