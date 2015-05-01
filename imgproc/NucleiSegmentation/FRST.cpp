#include "FRST.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

FRST::FRST() : 
  _S(NULL),
  _transform(TransformType::OrientationAndMagnitude),
  _symmetry(SymmetryType::DarkAndLight)
{
  _S = new Mat();
}

FRST::~FRST() {
  if (_S) {
    delete _S;
    _S = NULL;
  }
}

void FRST::frst2D(const Mat& image, Mat& S, const std::vector<float>& radii, const unsigned int& alpha, const float& beta, std::vector<float>& kappa)
{
	if (kappa.empty()) {
    kappa.push_back(9.9);
  }

  int rows=image.rows;
	int cols=image.cols;
 
  Mat gradX = Mat();
  Mat gradY = Mat();
  Mat magni = Mat();
       
  // Determine gradient magnitude and unit gradient
  Sobel(image, gradX, CV_64F, 1, 0, 3, 1, 0, BORDER_REFLECT);
  Sobel(image, gradY, CV_64F, 0, 1, 3, 1, 0, BORDER_REFLECT);
  magnitude(gradX, gradY, magni);

  double* gradXPtr = gradX.ptr<double>(0);
  double* gradYPtr = gradY.ptr<double>(0);
  double* magniPtr = magni.ptr<double>(0);
  double maxMagnitude = 0;
  for (int i = 0; i < (gradX.rows * gradX.cols); ++i) {
    *(gradXPtr+i) = *(magniPtr+i) > 0 ? *(gradXPtr+i) / *(magniPtr+i) : 0;
    *(gradYPtr+i) = *(magniPtr+i) > 0 ? *(gradYPtr+i) / *(magniPtr+i) : 0;
    maxMagnitude = *(magniPtr+i) > maxMagnitude ? *(magniPtr+i) : maxMagnitude;
  }

  double magniThreshold = maxMagnitude * beta;

  // Compute actual transform (On and Mn)
  S = Mat(rows, cols, CV_64F, 0.0);
  for (std::vector<float>::const_iterator it = radii.begin(); it != radii.end(); ++it) {
    float currentRadius = *it;
    Mat On = Mat(rows, cols, CV_64F, 0.0);
    Mat Mn = Mat(rows, cols, CV_64F, 0.0);
    Mat Fn = Mat(rows, cols, CV_64F, 0.0);
    Mat Sn = Mat(rows, cols, CV_64F, 0.0);

    double* gradXPtr = gradX.ptr<double>(0);
    double* gradYPtr = gradY.ptr<double>(0);
    double* magniPtr = magni.ptr<double>(0);
    double* OnPtr = On.ptr<double>(0);
    double* MnPtr = Mn.ptr<double>(0);

    // Fill On and Mn
    unsigned int rowStep = gradX.step1();
    for (int y = 0; y < rows; ++y) {
      unsigned int curRowStep = y * rowStep;
      for (int x = 0; x < cols; ++x) {
        if (*(magniPtr  +  + curRowStep + x) < magniThreshold) {
          continue;
        }
        if (_symmetry == SymmetryType::OnlyLight || _symmetry == SymmetryType::DarkAndLight) {
          int posX = x + floor(0.5 + currentRadius * *(gradXPtr + curRowStep + x));
          int posY = y + floor(0.5 + currentRadius * *(gradYPtr + curRowStep + x));
          posX = posX >= 0 ? posX : 0; 
          posY = posY >= 0 ? posY : 0; 
          posX = posX < cols ? posX : cols - 1;
          posY = posY < rows ? posY : rows - 1;
          int posInd = rowStep*posY + posX;
          *(OnPtr + posInd) += 1;
          *(MnPtr + posInd) += *(magniPtr + posInd);
        }

        if (_symmetry == SymmetryType::OnlyDark || _symmetry == SymmetryType::DarkAndLight) {
          int negX = x - floor(0.5 + currentRadius * *(gradXPtr + curRowStep + x));
          int negY = y - floor(0.5 + currentRadius * *(gradYPtr + curRowStep + x));
          negX = negX >= 0 ? negX : 0; 
          negY = negY >= 0 ? negY : 0; 
          negX = negX < cols ? negX : cols - 1;
          negY = negY < rows ? negY : rows - 1;
          int negInd = rowStep*negY + negX;
          *(OnPtr + negInd) -= 1;
          *(MnPtr + negInd) -= *(magniPtr + negInd);
        }
      }
    }

    // Create Fn
    unsigned int radiusInd = it - radii.begin();
    float currentKappa = kappa.size() > radiusInd ? kappa[radiusInd] : kappa[kappa.size() - 1];
    double* FnPtr = Fn.ptr<double>(0);
    for (int i = 0; i < (On.rows * On.cols); ++i) {
      double OnTilde = *(OnPtr+i) < currentKappa ? *(OnPtr+i) : currentKappa;
      if (_transform == TransformType::OrientationAndMagnitude) {
        *(FnPtr+i) =  ((*(MnPtr+i)) / currentKappa)  * pow(OnTilde/currentKappa,alpha);
      } else {
        int OnSign = (0. < OnTilde) - (OnTilde < 0.);
        *(FnPtr+i) = OnSign * pow(abs(OnTilde)/currentKappa,alpha);
      }
    }
    float ks = (int)currentRadius % 2 == 0 ? currentRadius + 1 : currentRadius;
		GaussianBlur(Fn, Sn, Size(ks,ks), ks/4., ks/4., BORDER_REFLECT);
    S += Sn;
  }
  S = S/radii.size();
};


