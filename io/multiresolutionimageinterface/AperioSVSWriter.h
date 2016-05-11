//---------------------------------------------------------------------------
#ifndef _AperioSVSWriter
#define _AperioSVSWriter
#include "multiresolutionimageinterface_export.h"
#include "core/PathologyEnums.h"
#include "MultiResolutionImageWriter.h"
#include <string>
#include <vector>


//! This class can be used to write images to disk in a multi-resolution pyramid fashion.
//! It supports writing the image in parts, to facilitate processing pipelines or in one go,
//! in the first setting one should first open the file (openFile), then write the image
//! information (writeImageInformation), write the base parts (writeBaseParts) and then finish
//! the pyramid (finishImage). The class also contains a convenience function (writeImage), 
//! which writes an entire MultiResolutionImage to disk using the image properties (color, data)
//! and the specified codec.

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT AperioSVSWriter : public MultiResolutionImageWriter {
private:
  template <typename T> void writeThumbnail();

public:
  int finishImage();
  void setSpacing(std::vector<double>& spacing);
};

#endif
