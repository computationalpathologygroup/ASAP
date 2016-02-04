#ifndef _OpenSlideImage
#define _OpenSlideImage

#include "MultiResolutionImage.h"
#include "config/pathology_config.h"

struct _openslide;
typedef struct _openslide openslide_t;

class EXPORT_MULTIRESOLUTIONIMAGEINTERFACE OpenSlideImage : public MultiResolutionImage {

public:

  OpenSlideImage();
  ~OpenSlideImage();  
  bool initialize(const std::string& imagePath);
  double getMinValue(int channel = -1) { return 0.; }
  double getMaxValue(int channel = -1) { return 255.; }

  const bool getIgnoreAlpha() const;
  void setIgnoreAlpha(const bool ignoreAlpha);

  std::string getOpenSlideProperty(const std::string& propertyName);
  std::string getOpenSlideErrorState();

  void setCacheSize(const unsigned long long cacheSize);

protected :
  void cleanup();
  
  void* readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level);

  openslide_t* _slide;

private:

  bool _ignoreAlpha;
  std::string _errorState;

};

#endif