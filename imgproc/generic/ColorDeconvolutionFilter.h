#ifndef _ColorDeconvolutionFilter
#define _ColorDeconvolutionFilter

#include <map>
#include <string>
#include <vector>
#include "ImageFilter.h"

template <class inType>  
class ColorDeconvolutionFilter :  public ImageFilter<inType, double> {

  bool checkInputImageRequirements(const Patch<inType>& input) const {
    bool validInput = true;
    validInput &= (input.getDimensions().size() == 3);
    validInput &= (input.getColorType() == pathology::ColorType::RGB || input.getColorType() == pathology::ColorType::ARGB);
    return validInput;
  }

  bool calculate(const Patch<inType>& input, Patch<double>& output) {
    std::vector<unsigned long long> dims = input.getDimensions();
    dims[2] = 1;
    output = Patch<double>(dims, pathology::ColorType::Monochrome);
    const inType* inPtr = input.getPointer();
    double* outPtr = output.getPointer();
    // Clamp box of output image against image extent to avoid that unused areas are processed.
    // Process all voxels of the valid region of the output page.
    for (unsigned int y = 0; y < dims[0]; ++y) {
      for (unsigned int x = 0; x < dims[1]; ++x) {
        std::vector<double> rgb(3);
        if (input.getColorType() == pathology::ColorType::ARGB) {
          inPtr += 2;
          for (unsigned int c = 0; c < 3; ++c, --inPtr) {
            rgb[c] = *inPtr;
          }
          inPtr += 5;
        }
        else {
          for (unsigned int c = 0; c < 3; ++c, ++inPtr) {
            rgb[c] = *inPtr;
          }
        }
        if (rgb[0] != 0 && rgb[1] != 0 && rgb[2] != 0) {
          double Rlog = -log(rgb[0] / 255.0);
          double Glog = -log(rgb[1] / 255.0);
          double Blog = -log(rgb[2] / 255.0);
          double val = 0.0;
          if ((Rlog + Glog + Blog) / 3. > _globalThreshold && Rlog > _rgbThresholds[0] && Glog > _rgbThresholds[1] && Blog > _rgbThresholds[2]) {
            double Rscaled = Rlog * _q[_outputStain * 3];
            double Gscaled = Glog * _q[_outputStain * 3 + 1];
            double Bscaled = Blog * _q[_outputStain * 3 + 2];
            val = Rscaled + Gscaled + Bscaled;
            if (val < 0.0) {
              val = 0.0;
            }
            if (val > _maxVal) {
              _maxVal = val;
            }
          }
          *outPtr = val;
        }
        else {
          *outPtr = 0;
        }
        ++outPtr;
      }
      if (this->shouldCancel()) {
        this->updateProgress(100);
        return false;
      }
      else {
        this->updateProgress(100*y/static_cast<float>(dims[0]));
      }
    }
    this->updateProgress(100);
    return true;
  }

  void calculateDeconvolutionMatrix(std::vector<double>& q) {
    double cosx[3];
    double cosy[3];
    double cosz[3];
    double len[3];

    for (unsigned int i = 0; i<3; ++i){
      /* normalise vector length */
      cosx[i] = cosy[i] = cosz[i] = 0.0;
      len[i] = sqrt(_modX[i] * _modX[i] + _modY[i] * _modY[i] + _modZ[i] * _modZ[i]);
      if (len[i] != 0.0){
        cosx[i] = _modX[i] / len[i];
        cosy[i] = _modY[i] / len[i];
        cosz[i] = _modZ[i] / len[i];
      }
    }

    /* translation matrix */
    if (cosx[1] == 0.0){ /* 2nd colour is unspecified */
      if (cosy[1] == 0.0){
        if (cosz[1] == 0.0){
          cosx[1] = cosz[0];
          cosy[1] = cosx[0];
          cosz[1] = cosy[0];
        }
      }
    }

    if (cosx[2] == 0.0){ /* 3rd colour is unspecified */
      if (cosy[2] == 0.0){
        if (cosz[2] == 0.0){
          if ((cosx[0] * cosx[0] + cosx[1] * cosx[1])> 1)
            cosx[2] = 0.0;
          else
            cosx[2] = sqrt(1.0 - (cosx[0] * cosx[0]) - (cosx[1] * cosx[1]));

          if ((cosy[0] * cosy[0] + cosy[1] * cosy[1])> 1)
            cosy[2] = 0.0;
          else
            cosy[2] = sqrt(1.0 - (cosy[0] * cosy[0]) - (cosy[1] * cosy[1]));

          if ((cosz[0] * cosz[0] + cosz[1] * cosz[1])> 1)
            cosz[2] = 0.0;
          else
            cosz[2] = sqrt(1.0 - (cosz[0] * cosz[0]) - (cosz[1] * cosz[1]));
        }
      }
    }

    double leng = sqrt(cosx[2] * cosx[2] + cosy[2] * cosy[2] + cosz[2] * cosz[2]);

    cosx[2] = cosx[2] / leng;
    cosy[2] = cosy[2] / leng;
    cosz[2] = cosz[2] / leng;

		for (int i=0; i<3; i++){
			if (cosx[i] == 0.0) cosx[i] = 0.001;
			if (cosy[i] == 0.0) cosy[i] = 0.001;
			if (cosz[i] == 0.0) cosz[i] = 0.001;
		}

    /* matrix inversion */
    double A = cosy[1] - cosx[1] * cosy[0] / cosx[0];
    double V = cosz[1] - cosx[1] * cosz[0] / cosx[0];
    double C = cosz[2] - cosy[2] * V / A + cosx[2] * (V / A * cosy[0] / cosx[0] - cosz[0] / cosx[0]);
    q[2] = (-cosx[2] / cosx[0] - cosx[2] / A * cosx[1] / cosx[0] * cosy[0] / cosx[0] + cosy[2] / A * cosx[1] / cosx[0]) / C;
    q[1] = -q[2] * V / A - cosx[1] / (cosx[0] * A);
    q[0] = 1.0 / cosx[0] - q[1] * cosy[0] / cosx[0] - q[2] * cosz[0] / cosx[0];
    q[5] = (-cosy[2] / A + cosx[2] / A * cosy[0] / cosx[0]) / C;
    q[4] = -q[5] * V / A + 1.0 / A;
    q[3] = -q[4] * cosy[0] / cosx[0] - q[5] * cosz[0] / cosx[0];
    q[8] = 1.0 / C;
    q[7] = -q[8] * V / A;
    q[6] = -q[7] * cosy[0] / cosx[0] - q[8] * cosz[0] / cosx[0];
  }

  double _modX[3];
  double _modY[3];
  double _modZ[3];
  std::vector<double> _q;

  unsigned int _outputStain;
  double _globalThreshold;
  std::vector<double> _rgbThresholds;
  double _maxVal;

public: 

  double getMinValue(int channel = -1) { return 0.; }
  double getMaxValue(int channel = -1) { return _maxVal; }

  ColorDeconvolutionFilter() : 
    ImageFilter<inType, double>(),
    _q(9, 0.0),
    _outputStain(0),
    _globalThreshold(0.25),
    _rgbThresholds(std::vector<double>(3,0.2)),
    _maxVal(0.0)
  {
    this->_samplesPerPixel = 1;
    this->_colorType = pathology::Monochrome;
    revertToDefaultStain();
  };

  ~ColorDeconvolutionFilter() {};

  std::string name() const {return "ColorDeconvolutionFilter";};

  void setOutputStain(const unsigned int& outputStain) {
    _outputStain = outputStain;
  }

  unsigned int getOutputStain() const {
    return _outputStain;
  }

  void setRGBDensityThresholds(const std::vector<double>& thresholds) {
    _rgbThresholds = thresholds;
  }

  std::vector<double> getRGBDensityThresholds() const {
    return _rgbThresholds;
  }

  void setGlobalDensityThreshold(const double& threshold) {
    _globalThreshold = threshold;
  }

  double getGlobalDensityThreshold() const {
    return _globalThreshold;
  }

  void revertToDefaultStain()
  {
    /* GL Haem matrix */
    _modX[0] = 0.644211;
    _modY[0] = 0.716556;
    _modZ[0] = 0.266844;
    /* GL Eos matrix */
    _modX[1] = 0.092789;
    _modY[1] = 0.954111;
    _modZ[1] = 0.283111;
    /*  Zero matrix */
    _modX[2] = 0.0;
    _modY[2] = 0.0;
    _modZ[2] = 0.0;
    calculateDeconvolutionMatrix(this->_q);
  }

  std::vector<std::vector<double> > getStain() const {
    std::vector<std::vector<double> > rval(3, std::vector<double>(3,0.));
    for (unsigned int i = 0; i < 3; ++i) {
      rval[i][0] = _modX[i];
      rval[i][1] = _modY[i];
      rval[i][2] = _modZ[i];
    }
    return rval;
  }

  void setStain(const std::vector<double>& stain0, const std::vector<double>& stain1 = std::vector<double>(), const std::vector<double>& stain2 = std::vector<double>()) {
    if (stain0.empty() || stain0.size() != 3) {
      return;
    }
    for (unsigned int i = 0; i < stain0.size(); ++i) {
      _modX[0] = stain0[0];
      _modY[0] = stain0[1];
      _modZ[0] = stain0[2];
    }
    if (stain1.empty() || stain1.size() != 3) {
      _modX[1] = 0.;
      _modY[1] = 0.;
      _modZ[1] = 0.;
    }
    else {
      _modX[1] = stain1[0];
      _modY[1] = stain1[1];
      _modZ[1] = stain1[2];
    }
    if (stain2.empty() || stain2.size() != 3) {
      _modX[2] = 0.;
      _modY[2] = 0.;
      _modZ[2] = 0.;
    }
    else {
      _modX[2] = stain2[0];
      _modY[2] = stain2[1];
      _modZ[2] = stain2[2];
    }
    calculateDeconvolutionMatrix(this->_q);
  }
  
};

#endif