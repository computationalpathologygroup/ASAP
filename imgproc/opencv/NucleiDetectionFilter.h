#include "DIAGPathologyOpenCVBridge.h"
#include "imgproc/generic/FilterBase.h"
#include "imgproc/NucleiSegmentation/FRST.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "core/Point.h"
#include "core/ProgressMonitor.h"
#include <queue>

template <typename inType>
class NucleiDetectionFilter : public FilterBase {

  double _minVal;
  double _maxVal;
  float _hMaximaThreshold;
  float _minRadius;
  float _maxRadius;
  float _stepRadius;
  float _alpha;
  float _beta;
  unsigned int _nrOfDetectedNuclei;
  bool _monochromeInput;

  ColorDeconvolutionFilter<double>* _colorDeconvolutionFilter;

  cv::Mat hybridReconstruct(const cv::Mat& marker, const cv::Mat& mask)
  {
    cv::Mat paddedMask;
    cv::Mat paddedMarker;
    copyMakeBorder(mask, paddedMask, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    copyMakeBorder(marker, paddedMarker, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    // Forward pass
    const double* msk = paddedMask.ptr<double>(1) + 1;
    double* mar = paddedMarker.ptr<double>(1) + 1;
    for (unsigned int y = 1; y < paddedMarker.rows - 1; ++y) {
      for (unsigned int x = 1; x < paddedMarker.cols - 1; ++x) {
        const double neighborhood[5] = { *(mar - 1 - paddedMarker.cols), *(mar - paddedMarker.cols), *(mar - paddedMarker.cols + 1), *(mar - 1), *mar };
        double val = std::min(*std::max_element(neighborhood, neighborhood + 5), *msk);
        if (val != *mar) {
          *mar = val;
        }
        ++msk;
        ++mar;
      }
      msk += 2;
      mar += 2;
    }
    updateProgress(35);
    // Backward pass
    msk-=3;
    mar-=3;
    std::queue < unsigned int > fifo;
    const int neighborhoodStrides[8] = { -1 - paddedMarker.cols, -paddedMarker.cols, -paddedMarker.cols + 1, -1, 1 + paddedMask.cols, paddedMask.cols, paddedMask.cols - 1, 1 };
    for (unsigned int y = paddedMarker.rows - 2; y > 0; --y) {
      for (unsigned int x = paddedMarker.cols - 2; x > 0; --x) {
        const double neighborhood[5] = { *(mar + neighborhoodStrides[4]), *(mar + neighborhoodStrides[5]), *(mar + neighborhoodStrides[6]), *(mar + neighborhoodStrides[7]), *mar };
        const double neighborhoodMask[4] = { *(msk + neighborhoodStrides[4]), *(msk + neighborhoodStrides[5]), *(msk + neighborhoodStrides[6]), *(msk + neighborhoodStrides[7]) };
        double val = std::min(*std::max_element(neighborhood, neighborhood + 5), *msk);
        if (val != *mar) {
          *mar = val;
        }
        for (unsigned int n = 0; n < 4; ++n) {
          if (neighborhood[n] < *mar && neighborhood[n] < neighborhoodMask[n]) {
            fifo.push(y*paddedMarker.cols + x);
            break;
          }
        }
        --msk;
        --mar;
      }
      msk -= 2;
      mar -= 2;
    }
    updateProgress(60);
    //Process FIFO-queue
    msk = paddedMask.ptr<double>();
    mar = paddedMarker.ptr<double>();
    while (!fifo.empty()) {
      unsigned int p = fifo.front();
      fifo.pop();
      for (unsigned int n = 0; n < 8; ++n) {
        double pVal = *(mar + p);
        double nVal = *(mar + p + neighborhoodStrides[n]);
        double nMaskVal = *(msk + p + neighborhoodStrides[n]);
        if (nVal < pVal && nVal != nMaskVal) {
          *(mar + p + neighborhoodStrides[n]) = std::min(pVal, nMaskVal);
          fifo.push(p + neighborhoodStrides[n]);
        }
      }
    }
    cv::Rect crop(1, 1, paddedMarker.cols - 2, paddedMarker.rows - 2);
    return paddedMarker(crop).clone();
  }

  bool checkInputImageRequirements(const Patch<inType>& input) const 
  {
    bool validInput = true;
    validInput &= ((input.getDimensions().size() == 3 && (input.getColorType() == pathology::ColorType::RGB || input.getColorType() == pathology::ColorType::ARGB || input.getColorType() == pathology::ColorType::Monochrome)));
    return validInput;
  }

  bool calculate(const Patch<inType>& input, std::vector<Point>& output) {
    if (input.getDimensions().size() == 3 && input.getColorType() == pathology::ColorType::Monochrome) {
      _monochromeInput = true;
    }
    else {
      _monochromeInput = false;
    }
    updateProgress(5);
    Patch<double> outp;    
    std::vector<double> spacing = input.getSpacing();
    if (spacing.empty()) {
      spacing.push_back(1.);
      spacing.push_back(1.);
    }
    else if (spacing.size() == 1) {
      spacing.push_back(1.);
    }
    if (!_monochromeInput) {
      _colorDeconvolutionFilter->filter(input, outp);
    }
    else {
      outp = input;
    }
    if (shouldCancel()) {
      updateProgress(100);
      return false;
    }
    cv::Mat inp = patchToMat(outp);
    cv::Mat out;
    FRST frst;
    std::vector<float> radii;
    for (float i = _minRadius; i <= _maxRadius; i += _stepRadius) {
      radii.push_back(i/spacing[0]);
    }
    frst.setSymmetryType(FRST::OnlyLight);
    frst.setTransformType(FRST::OrientationOnly);
    frst.frst2D(inp, out, radii, _alpha, _beta);
    if (shouldCancel()) {
      updateProgress(100);
      return false;
    }
    updateProgress(10);

    // H-maxima
    cv::Mat marker = out - _hMaximaThreshold;
    cv::Mat result = (out - hybridReconstruct(marker, out)) >= _hMaximaThreshold;
    if (shouldCancel()) {
      updateProgress(100);
      return false;
    }
    updateProgress(80);

    // Connected components
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(result, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // Centers
    output = std::vector<Point>();
    for (std::vector<std::vector<cv::Point> >::const_iterator it = contours.begin(); it != contours.end(); ++it) {
      float x = 0, y = 0;
      for (std::vector<cv::Point>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2) {
        x += it2->x;
        y += it2->y;
      }
      output.push_back(Point(x / it->size(), y / it->size()));
    }
    _nrOfDetectedNuclei = output.size();
    updateProgress(100);
    return true;
  };

public :

  NucleiDetectionFilter() :
    FilterBase(),
    _hMaximaThreshold(0.01),
    _colorDeconvolutionFilter(NULL),
    _alpha(2.),
    _beta(0.01),
    _minRadius(1.5),
    _maxRadius(5),
    _stepRadius(1),
    _nrOfDetectedNuclei(0),
    _monochromeInput(false)
  {
    _colorDeconvolutionFilter = new ColorDeconvolutionFilter<double>();
  }

  ~NucleiDetectionFilter() {
    if (_colorDeconvolutionFilter) {
      delete _colorDeconvolutionFilter;
    }
  };

  ColorDeconvolutionFilter<double>* getColorDeconvolutionFilter() {
    return _colorDeconvolutionFilter;
  }

  unsigned int getNumberOfDetectedNuclei() {
    return _nrOfDetectedNuclei;
  }

  void setMaximumRadius(const float& maxRadius) {
    _maxRadius = maxRadius;
  }

  float getMaximumRadius() {
    return _maxRadius;
  }

  void setMinimumRadius(const float& minRadius) {
    _minRadius = minRadius;
  }

  float getMinimumRadius() {
    return _minRadius;
  }

  void setRadiusStep(const float& stepRadius) {
    _stepRadius = stepRadius;
  }

  float getRadiusStep() {
    return _stepRadius;
  }

  void setAlpha(const float& alpha) {
    _alpha = alpha;
  }

  float getAlpha() {
    return _alpha;
  }

  void setBeta(const float& beta) {
    _beta = beta;
  }

  float getBeta() {
    return _beta;
  }

  void setHMaximaThreshold(const float& threshold) {
    _hMaximaThreshold = threshold;
  }

  float getHMaximaThreshold() {
    return _hMaximaThreshold;
  }

  double getMinValue(int channel = -1) {
    return _minVal;
  }

  double getMaxValue(int channel = -1) {
    return _maxVal;
  }


  std::string name() const { return std::string("NucleiDetectionFilter"); };

  bool filter(const Patch<inType>& input, std::vector<Point>& output) {
    if (checkInputImageRequirements(input)) {
      start();
      bool result = calculate(input, output);
      finish();
      return result;
    }
    else {
      return false;
    }
  }
  /*
  bool filter(unsigned int width, unsigned int height, unsigned int channels, pathology::ColorType ctype, inType *data, QVariant& output) {
    std::vector<unsigned long long> dims;
    dims.push_back(width);
    dims.push_back(height);
    dims.push_back(channels);
    Patch<inType> input(dims, ctype, data, false);
    return filter(input, output);
  }
  */
};