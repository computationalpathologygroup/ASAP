#ifndef _DIAGPathologyOpenCVBridge
#define _DIAGPathologyOpenCVBridge

#include "opencv2/core/core.hpp"
#include "core/Patch.h"

//! Converts an OpenCV Mat to a Patch
template <typename T>
Patch<T> matToPatch(const cv::Mat& mat, bool copyData = false) {
  std::vector<unsigned long long> dims;
  dims.push_back(mat.rows);
  dims.push_back(mat.cols);
  dims.push_back(mat.channels());  
  pathology::ColorType ctype = pathology::ColorType::RGB;
  if (mat.channels() == 4) {
    ctype = pathology::ColorType::ARGB;
  }
  else if (mat.channels() == 1) {
    ctype = pathology::ColorType::Monochrome;
  }
  else if (mat.channels() > 4) {
    ctype = pathology::ColorType::Indexed;
  }

  Patch<T> output;
  if (copyData) {
    output = Patch<T>(dims, ctype);
    std::copy(mat.data, mat.data + mat.cols*mat.rows*mat.channels(), output.getPointer());
  }
  else {
    output = Patch<T>(dims, ctype, (T*)mat.data, copyData);
  }
  return output;
}


//! Converts a Patch to an OpenCV Mat
template <typename T>
cv::Mat patchToMat(const Patch<T>& patch, bool copyData = false) {
  std::vector<unsigned long long> dims = patch.getDimensions();
  cv::Mat output;
  if (dims.size() == 3) {
    if (copyData) {
      output = cv::Mat_<T>(dims[1], dims[0] * dims[2]).reshape(dims[2]);
      std::copy(patch.getPointer(), patch.getPointer() + patch.getBufferSize(), (T*)output.data);
    }
    else {
      T* data = const_cast<T*>(patch.getPointer());
      output = cv::Mat_<T>(dims[1], dims[0]*dims[2], data);
      output = output.reshape(dims[2]);
    }
  }
  return output;
}

#endif