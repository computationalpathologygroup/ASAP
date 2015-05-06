#ifndef PatchH
#define PatchH

#include <vector>

#include "core/ImageSource.h"
#include "config/pathology_config.h"

template<typename T>
class Patch : public ImageSource {

private:
  T* _buffer;
  unsigned long long _bufferSize;
  bool _ownData;
  std::vector<unsigned long long> _dimensions;
  std::vector<unsigned long long> _strides;
  void calculateStrides();

  void swap(Patch<T>& first, Patch<T>& second);
  friend void Patch<T>::swap(Patch<T>& first, Patch<T>& second);

public :

  Patch();
  ~Patch();
  Patch(const Patch& rhs);
  Patch& operator=(const Patch rhs);
  Patch(const std::vector<unsigned long long>& dimensions, const pathology::ColorType& ctype = pathology::ColorType::Monochrome, T* data = NULL, bool ownData = true);

  // Arithmetic operators
  Patch<T> operator*(const T& val);
  Patch<T>& operator*=(const T& val);
  Patch<T> operator/(const T& val);
  Patch<T>& operator/=(const T& val);
  Patch<T> operator+(const T& val);
  Patch<T>& operator+=(const T& val);
  Patch<T> operator-(const T& val);
  Patch<T>& operator-=(const T& val);

  const T* getPointer() const;
  T* getPointer();

  std::vector<unsigned long long> getStrides();

  double getMinValue(int channel = -1);
  double getMaxValue(int channel = -1);

  T getValue(const std::vector<unsigned long long>& index) const;
  void setValue(const std::vector<unsigned long long>& index, const T& value);
  void fill(const T& value);
  void setSpacing(const std::vector<double>& spacing);

  bool empty();

  const std::vector<unsigned long long> getDimensions() const;
  const int getSamplesPerPixel() const;
  const unsigned long long getBufferSize() const;

  const pathology::DataType getDataType() const;

};

#include "Patch.hpp"

#endif