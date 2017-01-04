#include "Box.h"
#include <algorithm>

Box::Box() :
  _start(),
  _size()
{
}

Box::Box(const unsigned long long& x, const unsigned long long& y, const unsigned long long& width, const unsigned long long& height) {
  _start.resize(2);
  _size.resize(2);
  _start[0] = x;
  _start[1] = y;
  _size[0] = width;
  _size[1] = height;
}

Box::Box(const unsigned long long& x, const unsigned long long& y, const unsigned long long& z, const unsigned long long& width, const unsigned long long& height, const unsigned long long& depth) {
  _start.resize(3);
  _size.resize(3);
  _start[0] = x;
  _start[1] = y;
  _start[2] = z;
  _size[0] = width;
  _size[1] = height;
  _size[2] = depth;
}

Box::Box(const std::vector<unsigned long long>& size) :
  _size(size)
{
  _start.clear();
  _start.resize(size.size(), 0);
}

Box::Box(const std::vector<unsigned long long>& start, const std::vector<unsigned long long>& size) :
  _start(start),
  _size(size)
{
  if (_start.size() != _size.size()) {
    _start.clear();
    _size.clear();
  }
}

const std::vector<unsigned long long>& Box::getSize() const {
  return _size;
}

const std::vector<unsigned long long>& Box::getStart() const {
  return _start;
}

bool Box::intersects(const Box &b) const {
    if (b.getSize().size() != _size.size() || b.getStart().size() != _start.size()) {
      return false;
    }
    for (std::vector<unsigned long long>::const_iterator it = _size.begin(), itb = b.getSize().begin(); it != _size.end(); ++it, ++itb) {
      if ((*it) <= 0 || (*itb) <= 0) {
        return false;
      }
    }

    for (int i = 0; i < _start.size(); ++i) {
      if (_start[i] + _size[i] <= b.getStart()[i] || _start[i] >= b.getStart()[i] + b.getSize()[i]) {
        return false;
      }
    }
    return true;
  }

Box Box::intersection(const Box& b) const {
  if (!intersects(b)) {
    return Box();
  }
  std::vector<unsigned long long> start(_start.size(),0), size(_start.size(),0);
  for (int i = 0; i < _start.size(); ++i) {
    start[i] = std::max(_start[i], b.getStart()[i]);
    unsigned long long end = std::min(_start[i] + _size[i], b.getStart()[i] + b.getSize()[i]);
    if(end > start[i])
      size[i] = end - start[i];
    else
      size[i] = 0;
  }

  return Box(start, size);
}
