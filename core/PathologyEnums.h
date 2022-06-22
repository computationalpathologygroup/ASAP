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
    bool relative = false;
  };

  extern CORE_EXPORT std::map<std::string, LUT> DefaultColorLookupTables;

  enum class ColorType {
    InvalidColorType,
    Monochrome,
    RGB,
    RGBA,
    Indexed
  };

  enum class DataType {
    InvalidDataType,
    UChar,
    UInt16,
    UInt32,
    Float
  };

  enum class Compression {
    RAW,
    JPEG,
    LZW,
    JPEG2000
  };

  enum class Interpolation {
    NearestNeighbor,
    Linear
  };

}

#endif
