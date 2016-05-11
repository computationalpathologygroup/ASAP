//---------------------------------------------------------------------------
#ifndef _MultiResolutionImageWriter
#define _MultiResolutionImageWriter
#include "multiresolutionimageinterface_export.h"
#include "core/PathologyEnums.h"
#include <string>
#include <vector>

struct tiff;
typedef struct tiff TIFF;

class MultiResolutionImage;
class ProgressMonitor;

//! This class can be used to write images to disk in a multi-resolution pyramid fashion.
//! It supports writing the image in parts, to facilitate processing pipelines or in one go,
//! in the first setting one should first open the file (openFile), then write the image
//! information (writeImageInformation), write the base parts (writeBaseParts) and then finish
//! the pyramid (finishImage). The class also contains a convenience function (writeImage), 
//! which writes an entire MultiResolutionImage to disk using the image properties (color, data)
//! and the specified codec.

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT MultiResolutionImageWriter {
protected:

  //! Reference to the file to be written
  TIFF *_tiff;

  //! Reference to a progress monitor, this object is not the owner!
  ProgressMonitor* _monitor;

  //! Tile size
  unsigned int _tileSize;

  //! Number of indexed colors (only for ColorType Indexed)
  unsigned int _numberOfIndexedColors;

  //! JPEG compression quality
  float _quality;

  //! Compression
  pathology::Compression _codec;

  //! Pyramid interpolation type
  pathology::Interpolation _interpolation;

  //! Data type
  pathology::DataType _dType;

  //! Color type;
  pathology::ColorType _cType;

  //! Pixel spacing (normally taken from original image, but can be overwritten or provided when not specified)
  std::vector<double> _overrideSpacing;

  //! Min and max values of the image that is written to disk
  double* _min_vals;
  double* _max_vals;

  //! Positions in currently opened file
  unsigned int _pos;

  //! Currently opened file path
  std::string _fileName;

  void setBaseTags(TIFF* levelTiff);
  void setPyramidTags(TIFF* levelTiff, const unsigned long long& width, const unsigned long long& hight);
  void setTempPyramidTags(TIFF* levelTiff, const unsigned long long& width, const unsigned long long& hight);
  template <typename T> void writePyramidLevel(TIFF* levelTiff, unsigned int levelwidth, unsigned int levelheight, unsigned int nrsamples);
  template <typename T> T* downscaleTile(T* inTile, unsigned int tileSize, unsigned int nrSamples);
  template <typename T> int writePyramidToDisk();
  template <typename T> int incorporatePyramid();
  void writeBaseImagePartToTIFFTile(void* data, unsigned int pos);

  //! Temporary storage for the levelFiles
  std::vector<std::string> _levelFiles;

public:
  MultiResolutionImageWriter();
  virtual ~MultiResolutionImageWriter();

  // Opens the file for writing and keeps handle
  int openFile(const std::string& fileName);

  const std::string getOpenFile() const {return _fileName;}
  
  //! Writes the image information like image size, tile size, color and data types
  int writeImageInformation(const unsigned long long& sizeX, const unsigned long long& sizeY);

  //! Write image functions for different data types. This function provides functionality to write parts
  //! of the input image, so it does not have to be loaded in memory fully, can be useful for testing or
  //! large processing pipelines.
  void writeBaseImagePart(void* data);
  void writeBaseImagePartToLocation(void* data, const unsigned long long& x, const unsigned long long& y);

  //! Convience function to write an entire MultiResolutionImage to disk
  void writeImageToFile(MultiResolutionImage* img, const std::string& fileName);

  //! Will close the base image and finish writing the image pyramid and optionally the thumbnail image.
  //! Subsequently the image will be closed.
  virtual int finishImage();

  //! Sets the compression
  void setCompression(const pathology::Compression& codec) 
  {_codec = codec;}

  //! Gets the compression
  const pathology::Compression getCompression() const 
  {return _codec;}

  //! Sets the interpolation
  void setInterpolation(const pathology::Interpolation& interpolation) 
  {_interpolation = interpolation;}

  //! Gets the interpolation
  const pathology::Interpolation getInterpolation() const 
  {return _interpolation;}

  //! Sets the datatype
  void setDataType(const pathology::DataType& dType) 
  {
    _dType = dType;
  }

  //! Gets the compression
  const pathology::DataType getDataType() const 
  {return _dType;}

  //! Sets the compression
  void setColorType(const pathology::ColorType& cType) 
  {
    _cType = cType;
  }

  //! Sets the compression
  void setNumberOfIndexedColors(const unsigned int numberOfIndexedColors) 
  {
    _numberOfIndexedColors = numberOfIndexedColors;
  }

  //! Sets the compression
  unsigned int getNumberOfIndexedColors() const
  {
    return _numberOfIndexedColors;
  }

  //! Gets the compression
  const pathology::ColorType getColorType() const 
  {return _cType;}

  void setTileSize(const unsigned int& tileSize) {
    _tileSize = tileSize;
  }

  // Sets the pixel spacing of the image
  virtual void setSpacing(std::vector<double>& spacing);

  const unsigned int getTileSize() const {
    return _tileSize;
  }

  const std::vector<double> getOverrideSpacing() {
    return _overrideSpacing;
  }

  void setOverrideSpacing(const std::vector<double>& spacing) {
    _overrideSpacing = spacing;
  }

  //! Set JPEG quality (default value = 30)
  const int setJPEGQuality(const float& quality) 
  {if (quality > 0 && quality < 100) {_quality = quality; return 0;} else {return -1;} }

  const float getJPEGQuality() const 
  {{return _quality;}} 

  void setProgressMonitor(ProgressMonitor* monitor);

};

#endif
