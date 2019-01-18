#ifndef _PathologyEnums
#define _PathologyEnums

#include <string>
#include <map>
#include "core_export.h"

namespace pathology {

  struct LUT {
    unsigned char colors[256][4];
    bool wrapAround;
  };

  extern CORE_EXPORT const std::map<std::string, LUT> ColorLookupTables;

  enum ColorType : int {
    InvalidColorType,
    Monochrome,
    RGB,
    ARGB,
    Indexed
  };

  enum DataType : int  {
    InvalidDataType,
    UChar,
    UInt16,
    UInt32,
    Float
  };

  enum Compression : int  {
    RAW,
    JPEG,
    LZW,
    JPEG2000
  };

  enum Interpolation : int  {
    NearestNeighbor,
    Linear
  };

}

#endif
