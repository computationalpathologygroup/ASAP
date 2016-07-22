#ifndef _VSIImage
#define _VSIImage

#include <vector>
#include "MultiResolutionImage.h"
#include "multiresolutionimageinterface_export.h"

class MULTIRESOLUTIONIMAGEINTERFACE_EXPORT VSIImage : public MultiResolutionImage {

public:
  VSIImage();
  ~VSIImage();

  bool initializeType(const std::string& imagePath);

protected :

  void cleanup();
  
  void* readDataFromImage(const long long& startX, const long long& startY, const unsigned long long& width, 
    const unsigned long long& height, const unsigned int& level);

  double getMinValue(int channel = -1) { return 0.; }
  double getMaxValue(int channel = -1) { return 255.; }

private :
	std::string _vsiFileName;
	std::string _etsFile;
	std::vector<unsigned long long> _tileOffsets;
	std::vector<std::vector<int> > _tileCoords;
	unsigned int _tileSizeX;
	unsigned int _tileSizeY;
	unsigned int _nrTilesX;
	unsigned int _nrTilesY;
  unsigned int _compressionType;

  char* decodeTile(int no, int row, int col) const;
  unsigned long long parseETSFile(std::ifstream& ets);
};

#endif