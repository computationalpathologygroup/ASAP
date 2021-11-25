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
class DcmElement;

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

    DcmElement* _opticalPathSequence;

    DcmImageType _imageType;
    TilingType _tiling;

    std::map<std::string, std::string> _UIDs;

    unsigned int _frameOffset;
    unsigned int _numberOfFrames;
    unsigned int _nrFocalFlanes;
    unsigned int _extendedDoFPlanes;
    double _extendedDoFPlaneDistance;
    std::string _focusMethod;
    unsigned int _width;
    unsigned int _height;
    double _widthInMm;
    double _heightInMm;
    unsigned short _tileHeight;
    unsigned short _tileWidth;
    unsigned short _samplesPerPixel;
    std::string _photometricInterpretation;
    unsigned int _instanceNumber;
    double _sliceThickness;
    

    bool _isValid;
 
};

#endif