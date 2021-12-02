#ifndef _DICOMImage
#define _DICOMImage

#include "MultiResolutionImage.h"
#include "dicomfileformat_export.h"
#include <vector>
#include <map>

class WSIDicomInstance;

class DICOMFILEFORMAT_EXPORT DICOMImage : public MultiResolutionImage {

public:

  DICOMImage();
  ~DICOMImage();
  bool initializeType(const std::string& imagePath);
  double getMinValue(int channel = -1) { return 0.; }
  double getMaxValue(int channel = -1) { return 255.; }

  std::string getProperty(const std::string& propertyName);
  
  void setCacheSize(const unsigned long long cacheSize);

protected :
  void cleanup();
  
  void* readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level);

private:
    std::vector<std::vector<WSIDicomInstance*> > _levels;
    WSIDicomInstance* _label;
    WSIDicomInstance* _overview;

};

#endif