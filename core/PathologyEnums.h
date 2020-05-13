#ifndef _PathologyEnums
#define _PathologyEnums

#include <string>
#include <map>
#include <vector>
#include <array>
#include "core_export.h"

typedef std::array<float, 4> rgbaArray;

namespace pathology {

  struct LUT {
    std::vector<float> indices;
    std::vector<rgbaArray> colors;
    bool wrapAround;
  };

  extern CORE_EXPORT std::map<std::string, LUT> DefaultColorLookupTables;

  enum ColorType : int {
    InvalidColorType,
    Monochrome,
    RGB,
    RGBA,
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
