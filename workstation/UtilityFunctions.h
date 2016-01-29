#include <QImage>
#include "core/PathologyEnums.h"

inline unsigned int applyLUT(const float& val, const std::string& LUTname) {
  const pathology::LUT& currentLUT = pathology::ColorLookupTables.at(LUTname);
  const unsigned char(&LUTcolors)[256][4] = currentLUT.colors;
  if (currentLUT.wrapAround) {
    int ind = static_cast<unsigned int>(val) % 256;
    const unsigned char(&currentColor)[4] = LUTcolors[ind];
    return qRgba(currentColor[0], currentColor[1], currentColor[2], currentColor[3]);
  }
  else {
    int ind = val;
    const unsigned char(&currentColor)[4] = LUTcolors[ind];
    return qRgba(currentColor[0], currentColor[1], currentColor[2], currentColor[3]);
  }
}

template<typename T>
QImage convertMonochromeToRGB(T* data, unsigned int width, unsigned int height, unsigned int channel, unsigned int numberOfChannels, double channelMin, double channelMax, const std::string& LUTname) {
  QImage img(width, height, QImage::Format_ARGB32_Premultiplied);

  // Access the image at low level.  From the manual, a 32-bit RGB image is just a
  // vector of QRgb (which is really just some integer typedef)
  QRgb *pixels = reinterpret_cast<QRgb*>(img.bits());

  for (unsigned int i = channel, j = 0; i < width*height*numberOfChannels; i += numberOfChannels, ++j)
  {
    T pixel_msb = data[i];
    if (channelMax > channelMin) {
      if (pixel_msb < channelMin) {
        pixel_msb = channelMin;
      }
      else if (pixel_msb > channelMax) {
        pixel_msb = channelMax;
      }
      pixel_msb = ((pixel_msb - channelMin) / (channelMax - channelMin))*255.;
      pixels[j] = applyLUT(pixel_msb, LUTname);
    }
  }
  return img;
}