#ifndef _DICOMIMAGEFACTORY
#define _DICOMIMAGEFACTORY

#include "dicomfileformat_export.h"
#include "MultiResolutionImageFactory.h"

class DICOMFILEFORMAT_EXPORT DICOMImageFactory : public MultiResolutionImageFactory {
public:
	DICOMImageFactory();
	~DICOMImageFactory();

private:
  MultiResolutionImage* readImage(const std::string& fileName) const;
  bool canReadImage(const std::string& fileName) const;
};

extern "C" {
	DICOMFILEFORMAT_EXPORT void filetypeLoad();
}

#endif