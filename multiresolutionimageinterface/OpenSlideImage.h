#ifndef _OpenSlideImage
#define _OpenSlideImage

#include "MultiResolutionImage.h"
#include "openslidefileformat_export.h"

struct _openslide;
typedef struct _openslide openslide_t;

class OPENSLIDEFILEFORMAT_EXPORT OpenSlideImage : public MultiResolutionImage {

public:

  OpenSlideImage();
  ~OpenSlideImage();  
  bool initializeType(const std::string& imagePath);
  double getMinValue(int channel = -1) { return 0.; }
  double getMaxValue(int channel = -1) { return 255.; }

  std::string getProperty(const std::string& propertyName);
  std::string getOpenSlideErrorState();

  void setCacheSize(const unsigned long long cacheSize);

protected :
  void cleanup();
  
  void* readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level);

  openslide_t* _slide;

private:

  std::string _errorState;

  unsigned char _bg_r;
  unsigned char _bg_g;
  unsigned char _bg_b;

};

#endif