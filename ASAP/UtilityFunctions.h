#include <QImage>
#include <array>
#include <tuple>
#include <math.h>
#include "core/PathologyEnums.h"


inline std::tuple<float, float, float> rgb2hsv(std::tuple<float, float, float> rgb)
{
    std::tuple<float, float, float> hsv;
    double min = std::get<0>(rgb) < std::get<1>(rgb) ? std::get<0>(rgb) : std::get<1>(rgb);
    min = min < std::get<2>(rgb) ? min : std::get<2>(rgb);

    double max = std::get<0>(rgb) > std::get<1>(rgb) ? std::get<0>(rgb) : std::get<1>(rgb);
    max = max > std::get<2>(rgb) ? max : std::get<2>(rgb);

    std::get<2>(hsv) = max;                                // v
    double delta = max - min;
    if (delta < 0.00001)
    {
        std::get<1>(hsv) = 0;
        std::get<0>(hsv) = 0; // undefined, maybe nan?
        return hsv;
    }
    if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
        std::get<1>(hsv) = (delta / max);                  // s
    }
    else {
        // if max is 0, then r = g = b = 0              
        // s = 0, h is undefined
        std::get<1>(hsv) = 0.0;
        std::get<0>(hsv) = NAN;                            // its now undefined
        return hsv;
    }
    if (std::get<0>(rgb) >= max)                           // > is bogus, just keeps compilor happy
        std::get<0>(hsv) = (std::get<1>(rgb) - std::get<2>(rgb)) / delta;        // between yellow & magenta
    else
        if (std::get<1>(rgb) >= max)
            std::get<0>(hsv) = 2.0 + (std::get<2>(rgb) - std::get<0>(rgb)) / delta;  // between cyan & yellow
        else
            std::get<0>(hsv) = 4.0 + (std::get<0>(rgb) - std::get<1>(rgb)) / delta;  // between magenta & cyan

    std::get<0>(hsv) *= 60.0;                              // degrees

    if (std::get<0>(hsv) < 0.0)
        std::get<0>(hsv) += 360.0;

    return hsv;
}


inline std::tuple<float, float, float> hsv2rgb(std::tuple<float, float, float> hsv)
{
    std::tuple<float, float, float> out;

    if (std::get<1>(hsv) <= 0.0) {       // < is bogus, just shuts up warnings
        std::get<0>(out) = std::get<2>(hsv);
        std::get<1>(out) = std::get<2>(hsv);
        std::get<2>(out) = std::get<2>(hsv);
        return out;
    }
    double  hh = std::get<0>(hsv);
    if (hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    long i = (long)hh;
    double ff = hh - i;
    double p = std::get<2>(hsv) * (1.0 - std::get<1>(hsv));
    double q = std::get<2>(hsv) * (1.0 - (std::get<1>(hsv) * ff));
    double t = std::get<2>(hsv) * (1.0 - (std::get<1>(hsv) * (1.0 - ff)));

    switch (i) {
    case 0:
        std::get<0>(out) = std::get<2>(hsv);
        std::get<1>(out) = t;
        std::get<2>(out) = p;
        break;
    case 1:
        std::get<0>(out) = q;
        std::get<1>(out) = std::get<2>(hsv);
        std::get<2>(out) = p;
        break;
    case 2:
        std::get<0>(out) = p;
        std::get<1>(out) = std::get<2>(hsv);
        std::get<2>(out) = t;
        break;

    case 3:
        std::get<0>(out) = p;
        std::get<1>(out) = q;
        std::get<2>(out) = std::get<2>(hsv);
        break;
    case 4:
        std::get<0>(out) = t;
        std::get<1>(out) = p;
        std::get<2>(out) = std::get<2>(hsv);
        break;
    case 5:
    default:
        std::get<0>(out) = std::get<2>(hsv);
        std::get<1>(out) = p;
        std::get<2>(out) = q;
        break;
    }
    return out;
}

inline unsigned int applyLUT(const float& val, const pathology::LUT& LUT) {
  const std::vector<float>& LUTindices = LUT.indices;
  const std::vector<rgbaArray >& LUTcolors = LUT.colors;
  if (LUTcolors.size() == 0 || LUTindices.size() == 0) {
    return qRgba(0,0,0,0);
  }
  float ind = val;
  auto larger = std::upper_bound(LUTindices.begin(), LUTindices.end(), ind); 
  rgbaArray currentColor;
  if (larger == LUTindices.begin()) {
          currentColor = LUTcolors[0];
  }
  else if (larger == LUTindices.end()) {
      currentColor = LUTcolors.back();
  }
  else if (val - 0.0001 <= *(larger - 1) && *(larger - 1) <= val + 0.0001) {
      currentColor = LUTcolors[(larger - LUTindices.begin()) - 1];
  }
  else {
      auto index_next = larger - LUTindices.begin();
      float index_next_val = *larger;
      float index_prev_val = *(larger - 1);
      float index_range = index_next_val - index_prev_val;
      float val_normalized = (val - index_prev_val) / index_range;
      rgbaArray rgba_prev = LUTcolors[index_next - 1];
      rgbaArray rgba_next = LUTcolors[index_next];
      std::tuple<float, float, float> rgb_prev = std::make_tuple(rgba_prev[0]/255., rgba_prev[1] / 255., rgba_prev[2] / 255.);
      std::tuple<float, float, float> rgb_next = std::make_tuple(rgba_next[0] / 255., rgba_next[1] / 255., rgba_next[2] / 255.);
      std::tuple<float, float, float> hsv_prev = rgb2hsv(rgb_prev);
      std::tuple<float, float, float> hsv_next = rgb2hsv(rgb_next);
      std::tuple<float, float, float> hsv_interp;
      std::get<0>(hsv_interp) = std::get<0>(hsv_prev) * (1 - val_normalized) + std::get<0>(hsv_next) * val_normalized;
      std::get<1>(hsv_interp) = std::get<1>(hsv_prev) * (1 - val_normalized) + std::get<1>(hsv_next) * val_normalized;
      std::get<2>(hsv_interp) = std::get<2>(hsv_prev) * (1 - val_normalized) + std::get<2>(hsv_next) * val_normalized;
      std::tuple<float, float, float> rgb_interp = hsv2rgb(hsv_interp);
      currentColor[0] = std::get<0>(rgb_interp) * 255;
      currentColor[1] = std::get<1>(rgb_interp) * 255;
      currentColor[2] = std::get<2>(rgb_interp) * 255;
      currentColor[3] = rgba_prev[3] * (1 - val_normalized) + rgba_next[3] * val_normalized;
    }
  float* currentColorBuffer = currentColor.data();
  return qRgba(*currentColorBuffer, *(currentColorBuffer + 1), *(currentColorBuffer + 2), *(currentColorBuffer + 3));
}

template<typename T>
QImage convertMonochromeToRGB(T* data, unsigned int width, unsigned int height, unsigned int channel, unsigned int numberOfChannels, double channelMin, double channelMax, const pathology::LUT& LUT) {
  QImage img(width, height, QImage::Format_ARGB32_Premultiplied);

  // Access the image at low level.  From the manual, a 32-bit RGB image is just a
  // vector of QRgb (which is really just some integer typedef)
  std::map<T, QRgb> valToQrgb;
  QRgb *pixels = reinterpret_cast<QRgb*>(img.bits());
  for (unsigned int i = channel, j = 0; i < width*height*numberOfChannels; i += numberOfChannels, ++j)
  {
    T pixelValue = data[i];
    auto it = valToQrgb.find(pixelValue);
    if (it == valToQrgb.end()) {
        QRgb colorForVal;
        if (LUT.relative) {
            colorForVal = applyLUT((pixelValue - channelMin) / (channelMax - channelMin), LUT);
        }
        else {
            colorForVal = applyLUT(pixelValue, LUT);
        }
      pixels[j] = colorForVal;
      valToQrgb[pixelValue] = colorForVal;
    }
    else {
      pixels[j] = it->second;
    }
  }
  return img;
}
