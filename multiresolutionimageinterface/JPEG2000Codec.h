#ifndef _JPEG2000Codec
#define _JPEG2000Codec
#include "jpeg2kcodec_export.h"

namespace pathology {
  enum class DataType;
  enum class ColorType;
}

class JPEG2KCODEC_EXPORT JPEG2000Codec
{  
public:
  JPEG2000Codec();
  ~JPEG2000Codec();

  void encode(char* data, unsigned int& size, const unsigned int& tileSize, const unsigned int& rate, const unsigned int& nrComponents, const pathology::DataType& dataType, const pathology::ColorType& colorSpace) const;
  void decode(unsigned char* buf, const unsigned int& inSize, const unsigned int& outSize);
  void decode(unsigned char* inBuf, const unsigned int& inSize, unsigned char* outBuf, const unsigned int& outSize);

};

#endif