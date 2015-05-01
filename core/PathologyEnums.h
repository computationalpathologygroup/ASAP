#ifndef _PathologyEnums
#define _PathologyEnums

namespace pathology {

  enum ColorType {
    InvalidColorType,
    Monochrome,
    RGB,
    ARGB,
    Indexed
  };

  enum DataType {
    InvalidDataType,
    UChar,
    UInt16,
    UInt32,
    Float
  };

  enum Compression {
    RAW,
    JPEG,
    LZW,
    JPEG2000_LOSSLESS,
    JPEG2000_LOSSY
  };

  enum Interpolation {
    NearestNeighbor,
    Linear
  };

}

#endif