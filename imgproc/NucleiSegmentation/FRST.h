#ifndef _FRST
#define _FRST
/* FRST   compute Fast Radial Symmetry Transform (FRST).
 *     Computes the Fast Radial Symmetry Transform (FRST) as described in 
 *       
 *       Loy & Zelinsky (2003), 
 *       Fast Radial Symmetry for Detecting Points of Interest,
 *       IEEE Transactions on Pattern Analysis and Machine Intelligence, 
 *       August 2003.
 *
 */

#include "nucleisegmentation_export.h"
#include <vector>

namespace cv {
  class Mat;
}

class NUCLEISEGMENTATION_EXPORT FRST {

public:
  FRST();
  virtual ~FRST();

  void frst2D(const cv::Mat& image, cv::Mat& S, const std::vector<float>& radii, const unsigned int& alpha = 2, const float& beta = 0.0, std::vector<float> kappa = std::vector<float>());

  enum TransformType {
    OrientationOnly,
    OrientationAndMagnitude
  };

  enum SymmetryType {
    OnlyDark,
    OnlyLight,
    DarkAndLight
  };

  TransformType getTransformType() {
    return _transform;
  }

  void setTransformType(const TransformType& transform) {
    _transform = transform;
  }

  SymmetryType getSymmetryType() {
    return _symmetry;
  }

  void setSymmetryType(const SymmetryType& symmetry) {
    _symmetry = symmetry;
  }

private :
   cv::Mat* _S;
   TransformType _transform;
   SymmetryType _symmetry;
};

#endif