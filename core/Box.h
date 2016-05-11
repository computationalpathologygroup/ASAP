#ifndef BoxH
#define BoxH

#include <vector>

#include "core_export.h"

class CORE_EXPORT Box {
private :	  
  std::vector<unsigned long long> _start;
  std::vector<unsigned long long> _size;

public :

	Box intersection(const Box& r) const;
	bool intersects(const Box& r) const;

  Box();
  Box(const unsigned long long& x, const unsigned long long& y, const unsigned long long& width, const unsigned long long& height);
  Box(const unsigned long long& x, const unsigned long long& y, const unsigned long long& z, const unsigned long long& width, const unsigned long long& height, const unsigned long long& depth);
  Box(const std::vector<unsigned long long>& size);
  Box(const std::vector<unsigned long long>& start, const std::vector<unsigned long long>& size);
  
  const std::vector<unsigned long long>& getSize() const;
  const std::vector<unsigned long long>& getStart() const;

};

#endif