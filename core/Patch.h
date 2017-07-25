#ifndef PatchH
#define PatchH

#include <vector>

#include "core/ImageSource.h"

template<typename T>
class Patch : public ImageSource {

private:
  T* _buffer;
  unsigned long long _bufferSize;
  bool _ownData;
  std::vector<unsigned long long> _dimensions;
  std::vector<unsigned long long> _strides;
  std::vector<double> _wsiMinValues;
  std::vector<double> _wsiMaxValues;
  void calculateStrides();

  void swap(Patch<T>& first, Patch<T>& second);

public :

  Patch();
  ~Patch();
  Patch(const Patch& rhs);
  Patch& operator=(const Patch rhs);
  Patch(const std::vector<unsigned long long>& dimensions, const pathology::ColorType& ctype = pathology::Monochrome, T* data = NULL, bool ownData = true, std::vector<double> wsiMinValues = std::vector<double>(), std::vector<double> wsiMaxValues = std::vector<double>());

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

  double getWSIMinValue(int channel = -1) const;
  double getWSIMaxValue(int channel = -1) const;

  double getMinValue(int channel = -1);
  double getMaxValue(int channel = -1);
  double getMinValue(int channel = -1) const;
  double getMaxValue(int channel = -1) const;

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