//---------------------------------------------------------------------------
#ifndef _VSIImageReader
#define _VSIImageReader
#include <string>
#include <vector>
#include <map>
#include "config/pathology_config.h"

class EXPORT_MULTIRESOLUTIONIMAGEINTERFACE VSIImageReader {

public:

  VSIImageReader();
  ~VSIImageReader();

  //! Opens the slide file and keeps a reference to it
  void open(const std::string& fileName);

  //! Closes the open file and clears the reference
  void close();

  //! Checks whether the instance has an open file
  const bool isOpen() const;

  //! Gets the dimensions of the base level of the pyramid
  const std::vector<unsigned long long> getDimensions() const;

  //! Gets and sets the pixel size
  float getPixelSizeX() {return _pixelSizeX;}
  float getPixelSizeY() {return _pixelSizeY;}
  void setPixelSizeX(float pixelSizeX) {_pixelSizeX = pixelSizeX;}
  void setPixelSizeY(float pixelSizeY) {_pixelSizeY = pixelSizeY;}


  //! Obtains pixel data for a requested region as int8 array. The user is responsible for allocating
  //! enough memory for the data to fit the array and clearing the memory
  void readRegion(const unsigned long long& startX, const unsigned long long& startY, const unsigned long long& width, 
  const unsigned long long& height, unsigned char* data) const;

};

#endif