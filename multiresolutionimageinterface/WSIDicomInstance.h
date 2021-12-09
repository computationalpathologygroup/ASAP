#ifndef _WSIDicomInstance
#define _WSIDicomInstance

#include "dicomfileformat_export.h"
#include <vector>
#include <string>
#include <map>

class DcmFileFormat;
class DcmMetaInfo;
class DcmDataset;
class DcmPixelData;
class DcmItem;
class JPEG2000Codec;

class DICOMFILEFORMAT_EXPORT WSIDicomInstance  {

public:

    WSIDicomInstance() = delete;
    WSIDicomInstance(DcmFileFormat* fileformat);
    ~WSIDicomInstance();

    enum class DcmImageType {
        InvalidImageType,
        Volume,
        Label,
        Overview
    };

    enum class TilingType {
        Full,
        Sparse
    };

    bool initialize(DcmFileFormat* fileFormat);
    std::string getUID(const std::string& UIDName) const;
    const std::map<std::string, std::string>& getUIDs() const;
    std::vector<double> getPixelSpacing() const;
    std::vector<unsigned int> getSize() const;
    std::vector<unsigned short> getTileSize() const;
    std::vector<unsigned short> getSizeInTiles() const;
    DcmImageType getImageType() const;
    bool valid() const;

    void* getFrame(const long long& x, const long long& y, const long long& z = 0, const long long& op = 0);
    

private:

    static const std::vector<std::string> SUPPORTED_TRANSFER_SYNTAX;

    DcmFileFormat* _fileFormat;
    DcmMetaInfo* _metaInfo;
    DcmDataset* _dataset;
    DcmPixelData* _pixels;

    DcmItem* _opticalPathSequence;

    DcmImageType _imageType;
    TilingType _tiling;

    JPEG2000Codec* _jp2Codec;

    std::map<std::string, std::string> _UIDs;

    unsigned int _frameOffset;
    unsigned int _numberOfFrames;
    bool _extendedDoF;
    unsigned short _extendedDoFPlanes;
    float _extendedDoFPlaneDistance;
    std::string _focusMethod;
    double _pixelSpacingX;
    double _pixelSpacingY;
    double _sliceSpacing;
    unsigned int _width;
    unsigned int _height;
    float _widthInMm;
    float _heightInMm;
    float _depthInMm;
    unsigned short _tileHeight;
    unsigned short _tileWidth;
    unsigned short _samplesPerPixel;
    unsigned short _bitsPerSample;
    std::string _photometricInterpretation;
    unsigned int _instanceNumber;
    float _sliceThickness;
    std::map<unsigned long, unsigned long> _positionToFrameIndex;
    
    bool _isValid;
 
};

#endif