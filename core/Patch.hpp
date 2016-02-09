#ifndef PatchHPP
#define PatchHPP

#include "Patch.h"
#include <algorithm>
#include <limits>

template<>
inline const pathology::DataType Patch<unsigned char>::getDataType() const {
  return pathology::UChar;
}

template<>
inline const pathology::DataType Patch<float>::getDataType() const {
  return pathology::Float;
}

template<>
inline const pathology::DataType Patch<unsigned int>::getDataType() const {
  return pathology::UInt32;
}

template<>
inline const pathology::DataType Patch<unsigned short>::getDataType() const {
  return pathology::UInt16;
}

template<>
inline const pathology::DataType Patch<double>::getDataType() const {
  return pathology::Float;
}

template<typename T>
const pathology::DataType Patch<T>::getDataType() const {
  return pathology::InvalidDataType;
}

template<typename T>
void Patch<T>::setSpacing(const std::vector<double>& spacing) {
  _spacing = spacing;
}

template<typename T>
Patch<T>::Patch() :
  ImageSource(),
  _bufferSize(0),
  _buffer(NULL),
  _ownData(true)
{
  _isValid = true;
}

template<typename T>
Patch<T>::~Patch()
{
  if (_buffer && _ownData) {
    delete[] _buffer;
    _buffer = NULL;
  }
}

template<typename T>
double Patch<T>::getMinValue(int channel) {
  double min = std::numeric_limits<double>::max();
  if (_buffer) {
    for (int i = 0; i < _bufferSize; ++i) {
      if (_buffer[i] < min) {
        min = _buffer[i];
      }
    }
  }
  return min;
}

template<typename T>
double Patch<T>::getMaxValue(int channel) {
  double max = std::numeric_limits<double>::min();
  if (_buffer) {
    for (int i = 0; i < _bufferSize; ++i) {
      if (_buffer[i] > max) {
        max = _buffer[i];
      }
    }
  }
  return max;
}


template<typename T>
const int Patch<T>::getSamplesPerPixel() const {
  if (!_dimensions.empty()) {
    return _dimensions.back();
  }
  else {
    return 0;
  }
}

template<typename T>
const unsigned long long Patch<T>::getBufferSize() const {
  return _bufferSize;
}

template<typename T>
Patch<T>::Patch(const std::vector<unsigned long long>& dimensions, const pathology::ColorType& colorType, T* data, bool ownData) :
   ImageSource(),
  _dimensions(dimensions),
  _buffer(data),
  _ownData(ownData)
{
  _colorType = colorType;
  if (_dimensions.empty()) {
    _bufferSize = 0;
  } else {
    _bufferSize = 1;
  }
  for (int i = 0; i < _dimensions.size(); ++i) {
    _bufferSize *= _dimensions[i];
  }
  if (data && _bufferSize) {
    _buffer = data;
  }
  else if (_bufferSize) {
    _buffer = new T[_bufferSize];
  }
  if (!dimensions.empty()) {
    if ((_colorType == pathology::ARGB && dimensions.back() != 4) || (_colorType == pathology::RGB && dimensions.back() != 3) || (_colorType == pathology::Monochrome && dimensions.back() != 1)) {
      _colorType == pathology::Indexed;
    }
  }
  calculateStrides();
  _isValid = true;
}

template<typename T>
Patch<T>::Patch(const Patch<T>& rhs) : 
  ImageSource(rhs),
  _bufferSize(rhs._bufferSize),
  _dimensions(rhs._dimensions),
  _ownData(true),
  _strides(rhs._strides),
  _buffer(NULL)
{
  _buffer = new T[_bufferSize];
  std::copy(rhs._buffer, rhs._buffer + rhs._bufferSize, _buffer);
  _isValid = true;
}

template<typename T>
void Patch<T>::swap(Patch<T>& first, Patch<T>& second) {
  ImageSource::swap(first, second);
  std::swap(first._dimensions, second._dimensions);
  std::swap(first._buffer, second._buffer);
  std::swap(first._bufferSize, second._bufferSize);
  std::swap(first._ownData, second._ownData);
  std::swap(first._strides, second._strides);
}

template<typename T>
Patch<T>& Patch<T>::operator=(Patch<T> rhs) {
  this->swap(*this, rhs);
  return *this;
}

template<typename T>
const std::vector<unsigned long long> Patch<T>::getDimensions() const {
  return _dimensions;
}

template<typename T>
const T* Patch<T>::getPointer() const {
  return _buffer;
}

template<typename T>
T* Patch<T>::getPointer() {
  return _buffer;
}

template<typename T>
void Patch<T>::calculateStrides() {
  _strides.clear();
  _strides.resize(_dimensions.size(),1);
  if (!_strides.empty()) {
    for (int i = _strides.size() - 2; i >= 0; --i) {
    _strides[i] = _strides[i + 1] * _dimensions[i + 1];
    }
  }
}

template<typename T>
std::vector<unsigned long long> Patch<T>::getStrides() {
  return _strides;
}

template<typename T>
T Patch<T>::getValue(const std::vector<unsigned long long>& index) const {
  unsigned long long offset = 0;
  for (int i = 0; i < index.size(); ++i) {
    offset += index[i]*_strides[i];
  }
  return _buffer[offset];
}

template<typename T>
void Patch<T>::setValue(const std::vector<unsigned long long>& index, const T& value) {
  unsigned long long offset = 0;
  for (int i = 0; i < index.size(); ++i) {
    offset += index[i]*_strides[i];
  }
  _buffer[offset] = value;
}

template<typename T>
void Patch<T>::fill(const T& value) {
  std::fill(_buffer, _buffer + _bufferSize, value);
}

template<typename T>
Patch<T> Patch<T>::operator*(const T& val) {
  Patch<T> temp(*this);
  temp *= val;
  return temp;
}

template<typename T>
Patch<T>& Patch<T>::operator*=(const T& val) {
  T* ptr = this->getPointer();
  for (unsigned long long i = 0; i < _bufferSize; ++i) {
    (*ptr) *= val;
    ++ptr;
  }
  return *this;
}

template<typename T>
Patch<T> Patch<T>::operator/(const T& val) {
  Patch<T> temp(*this);
  temp /= val;
  return temp;
}

template<typename T>
Patch<T>& Patch<T>::operator/=(const T& val) {
  T* ptr = this->getPointer();
  for (unsigned long long i = 0; i < _bufferSize; ++i) {
    (*ptr) /= val;
    ++ptr;
  }
  return *this;
}

template<typename T>
Patch<T> Patch<T>::operator+(const T& val) {
  Patch<T> temp(*this);
  temp += val;
  return temp;
}

template<typename T>
Patch<T>& Patch<T>::operator+=(const T& val) {
  T* ptr = this->getPointer();
  for (unsigned long long i = 0; i < _bufferSize; ++i) {
    (*ptr) += val;
    ++ptr;
  }
  return *this;
}

template<typename T>
Patch<T> Patch<T>::operator-(const T& val) {
  Patch<T> temp(*this);
  temp -= val;
  return temp;
}

template<typename T>
Patch<T>& Patch<T>::operator-=(const T& val) {
  T* ptr = this->getPointer();
  for (unsigned long long i = 0; i < _bufferSize; ++i) {
    (*ptr) -= val;
    ++ptr;
  }
  return *this;
}

template<typename T>
bool Patch<T>::empty() {
  return _bufferSize == 0;
}

#endif