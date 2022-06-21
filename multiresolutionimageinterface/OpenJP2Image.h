#ifndef _OPENJP2IMAGE
#define _OPENJP2IMAGE

#include "MultiResolutionImage.h"
#include "multiresolutionimageinterface_export.h"

typedef void* opj_stream_t;

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT OpenJP2Image : public MultiResolutionImage {

public:

  OpenJP2Image();
  ~OpenJP2Image();
  bool initializeType(const std::string& imagePath);

  double getMinValue(int channel = -1);
  double getMaxValue(int channel = -1);
  long long getEncodedTileSize(const long long& startX, const long long& startY, const unsigned int& level);
  unsigned char* readEncodedDataFromImage(const long long& startX, const long long& startY, const unsigned int& level);

protected :
  void cleanup();
  
  void* readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level);

  template <typename T> T* FillRequestedRegion(const  long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level, unsigned int nrSamples);

  std::vector<std::vector<unsigned int> > _tileSizesPerLevel;

  std::vector<double> _minValues;
  std::vector<double> _maxValues;

  opj_stream_t* _stream;

};

#endif