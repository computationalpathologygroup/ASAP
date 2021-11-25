#ifndef _WSIDicomInstance
#define _WSIDicomInstance

#include "dicomfileformat_export.h"
#include <vector>
#include <string>
#include <map>

class DcmFileFormat;
class DcmMetaInfo;
class DcmDataset;
class DicomImage;
class DcmItem;

class DICOMFILEFORMAT_EXPORT WSIDicomInstance  {

public:

    WSIDicomInstance();
    ~WSIDicomInstance();

    bool initialize(DcmFileFormat* fileFormat);

private:

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

    static const std::vector<std::string> SUPPORTED_TRANSFER_SYNTAX;

    DcmFileFormat* _fileFormat;
    DcmMetaInfo* _metaInfo;
    DcmDataset* _dataset;
    DicomImage* _image;

    DcmItem* _opticalPathSequence;

    DcmImageType _imageType;
    TilingType _tiling;

    std::map<std::string, std::string> _UIDs;

    unsigned int _frameOffset;
    unsigned int _numberOfFrames;
    bool _extendedDoF;
    unsigned short _extendedDoFPlanes;
    float _extendedDoFPlaneDistance;
    std::string _focusMethod;
    unsigned int _width;
    unsigned int _height;
    float _widthInMm;
    float _heightInMm;
    float _depthInMm;
    unsigned short _tileHeight;
    unsigned short _tileWidth;
    unsigned short _samplesPerPixel;
    std::string _photometricInterpretation;
    unsigned int _instanceNumber;
    float _sliceThickness;
    

    bool _isValid;
 
};

#endif