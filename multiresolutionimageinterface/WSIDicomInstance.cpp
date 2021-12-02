#include "WSIDicomInstance.h"

#include <vector>

#include "dcmtk/config/osconfig.h"
#include "dcmtk/ofstd/ofcond.h"
#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcuid.h"
#include "dcmtk/dcmdata/dcfilefo.h"
#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmimage/diregist.h"
#include "dcmtk/dcmdata/dcxfer.h"
#include "dcmtk/dcmdata/dcmetinf.h"
#include "dcmtk/dcmdata/dcrlerp.h"
#include "dcmtk/dcmdata/dcrledrg.h"

// JPEG en/decoding
#include "dcmtk/dcmjpeg/djdecode.h"
#include "dcmtk/dcmjpeg/djencode.h"

const std::vector<std::string> WSIDicomInstance::SUPPORTED_TRANSFER_SYNTAX = { UID_JPEGProcess1TransferSyntax, UID_JPEGProcess2_4TransferSyntax, UID_JPEG2000LosslessOnlyTransferSyntax, UID_JPEG2000TransferSyntax };

WSIDicomInstance::WSIDicomInstance(DcmFileFormat* fileFormat) :
    _fileFormat(fileFormat), _dataset(fileFormat->getDataset()), _metaInfo(fileFormat->getMetaInfo()), _pixels(nullptr), _isValid(false),
    _frameOffset(0), _numberOfFrames(0), _imageType(DcmImageType::InvalidImageType), _tiling(TilingType::Sparse),
    _depthInMm(0), _widthInMm(0), _heightInMm(0), _opticalPathSequence(nullptr), _focusMethod(""), _extendedDoF(false),
    _extendedDoFPlaneDistance(0), _extendedDoFPlanes(0), _height(0), _width(0), _tileWidth(0), _tileHeight(0), _samplesPerPixel(0),
    _instanceNumber(0), _sliceThickness(0), _pixelSpacingX(0), _pixelSpacingY(0), _sliceSpacing(0), _bitsPerSample(0)
{
    DJDecoderRegistration::registerCodecs();
    DcmElement* element = NULL;
    if (_dataset->findAndGetElement(DCM_PixelData, element).good()) {
        _pixels = OFstatic_cast(DcmPixelData*, element);
    }

    OFString msSOPClassUID;
    std::vector<DcmTagKey> requiredTags = { DCM_StudyInstanceUID, DCM_SeriesInstanceUID, DCM_FrameOfReferenceUID, DCM_Rows, DCM_Columns,
                                            DCM_SamplesPerPixel, DCM_PhotometricInterpretation, DCM_ImageType, DCM_TotalPixelMatrixColumns,
                                            DCM_TotalPixelMatrixRows, DCM_NumberOfFrames, DCM_SharedFunctionalGroupsSequence, DCM_OpticalPathSequence };
    this->_metaInfo->findAndGetOFString(DCM_MediaStorageSOPClassUID, msSOPClassUID);
    if (msSOPClassUID == UID_VLWholeSlideMicroscopyImageStorage) {
        DcmXfer transferSyntax = _dataset->getOriginalXfer();
        if (std::find(this->SUPPORTED_TRANSFER_SYNTAX.begin(), this->SUPPORTED_TRANSFER_SYNTAX.end(), transferSyntax.getXferID()) == this->SUPPORTED_TRANSFER_SYNTAX.end()) {
            _isValid = false;
            return;
        }
        DcmElement* tagValue;
        for (auto tag : requiredTags) {
            if (this->_dataset->findAndGetElement(tag, tagValue).bad()) {
                _isValid = false;
                return;
            }
        }

        // Check the Image Type
        OFString imageType;
        this->_dataset->findAndGetOFString(DCM_ImageType, imageType, 2);
        if (imageType == "VOLUME") {
            _imageType = DcmImageType::Volume;
        }
        else if (imageType == "LABEL") {
            _imageType = DcmImageType::Label;
        }
        else if (imageType == "OVERVIEW") {
            _imageType = DcmImageType::Overview;
        }
        else {
            _imageType = DcmImageType::InvalidImageType;
            _isValid;
            return;
        }

        // Store the UIDs
        OFString uid;
        this->_dataset->findAndGetOFString(DCM_StudyInstanceUID, uid);
        this->_UIDs["StudyInstanceUID"] = std::string(uid.c_str());
        this->_dataset->findAndGetOFString(DCM_SeriesInstanceUID, uid);
        this->_UIDs["SeriesInstanceUID"] = std::string(uid.c_str());
        this->_dataset->findAndGetOFString(DCM_FrameOfReferenceUID, uid);
        this->_UIDs["FrameOfReferenceUID"] = std::string(uid.c_str());
        this->_dataset->findAndGetOFString(DCM_SOPInstanceUID, uid);
        this->_UIDs["SOPInstanceUID"] = std::string(uid.c_str());

        // Check whether this is a standalone image or part of a group of concatenated images
        if (this->_dataset->findAndGetOFString(DCM_SOPInstanceUIDOfConcatenationSource, uid).bad()) {
            this->_UIDs["SOPInstanceUIDOfConcatenationSource"] = "";
            this->_frameOffset = 0;
        }
        else {
            this->_UIDs["SOPInstanceUIDOfConcatenationSource"] = std::string(uid.c_str());
            if (this->_dataset->findAndGetUint32(DCM_ConcatenationFrameOffsetNumber, this->_frameOffset).bad()) {
                _isValid = false;
                return;
            }
        }
        OFString nrFrames;
        if (this->_dataset->findAndGetOFString(DCM_NumberOfFrames, nrFrames).good()) {
            this->_numberOfFrames = std::stoi(nrFrames);
        } else {
            this->_numberOfFrames = 1;
        }

        // Check the tiling
        this->_tiling = TilingType::Sparse;
        OFString tiling;
        if (this->_dataset->findAndGetOFString(DCM_DimensionOrganizationType, tiling).good()) {
            if (tiling == "TILED_FULL") {
                this->_tiling = TilingType::Full;
            }
        }

        DcmItem* sfgSeq;
        if (this->_dataset->findAndGetSequenceItem(DCM_SharedFunctionalGroupsSequence, sfgSeq).good()) {
            DcmItem* pmSeq;
            if (sfgSeq->findAndGetSequenceItem(DCM_PixelMeasuresSequence, pmSeq).good()) {
                pmSeq->findAndGetFloat64(DCM_PixelSpacing, _pixelSpacingX, 0);
                pmSeq->findAndGetFloat64(DCM_PixelSpacing, _pixelSpacingY, 1);
                pmSeq->findAndGetFloat64(DCM_SpacingBetweenSlices, _sliceSpacing);
                if (_pixelSpacingX == 0 || _pixelSpacingY == 0) {
                    _isValid = false;
                    return;
                }
            }
        }

        // Get focusing info
        OFString eDoF;
        if (this->_dataset->findAndGetOFString(DCM_ExtendedDepthOfField, eDoF).good()) {
            if (eDoF == "YES") {
                _extendedDoF = true;
                if (this->_dataset->findAndGetUint16(DCM_NumberOfFocalPlanes, _extendedDoFPlanes).bad()) {
                    _extendedDoFPlanes = 0;
                }
                if (this->_dataset->findAndGetFloat32(DCM_DistanceBetweenFocalPlanes, _extendedDoFPlaneDistance).bad()) {
                    _extendedDoFPlaneDistance = 0;
                }
            }
            else {
                _extendedDoF = false;
            }
        }
        OFString focusMethod = "";
        this->_dataset->findAndGetOFString(DCM_FocusMethod, focusMethod);
        _focusMethod = std::string(focusMethod.c_str());

        this->_dataset->findAndGetUint32(DCM_TotalPixelMatrixRows, _height);
        this->_dataset->findAndGetUint32(DCM_TotalPixelMatrixColumns, _width);
        if (_height == 0 || _width == 0) {
            _isValid = false;
            return;
        }

        this->_dataset->findAndGetFloat32(DCM_ImagedVolumeHeight, _heightInMm);
        this->_dataset->findAndGetFloat32(DCM_ImagedVolumeWidth, _widthInMm);
        this->_dataset->findAndGetFloat32(DCM_ImagedVolumeDepth, _depthInMm);
        this->_dataset->findAndGetUint16(DCM_Rows, _tileHeight);
        this->_dataset->findAndGetUint16(DCM_Columns, _tileWidth);
        this->_dataset->findAndGetUint16(DCM_SamplesPerPixel, _samplesPerPixel);
        OFString photometricInterpretation = "";
        this->_dataset->findAndGetOFString(DCM_PhotometricInterpretation, photometricInterpretation);
        _photometricInterpretation = std::string(photometricInterpretation.c_str());
        this->_dataset->findAndGetUint32(DCM_InstanceNumber, _instanceNumber);
        if (this->_dataset->findAndGetFloat32(DCM_SliceThickness, _sliceThickness).bad()) {
            _sliceThickness = _depthInMm;
        }
        this->_dataset->findAndGetSequenceItem(DCM_OpticalPathSequence, _opticalPathSequence);

        DcmItem* pfSeq;
        if (this->_tiling == TilingType::Sparse) {
            for (long i = 0; i < this->_numberOfFrames; ++i) {
                if (this->_dataset->findAndGetSequenceItem(DCM_PerFrameFunctionalGroupsSequence, pfSeq, i).good()) {
                    DcmItem* ppsSeq;
                    if (pfSeq->findAndGetSequenceItem(DCM_PlanePositionSlideSequence, ppsSeq).good()) {
                        Sint32 colPos = 0;
                        Sint32 rowPos = 0;
                        ppsSeq->findAndGetSint32(DCM_ColumnPositionInTotalImagePixelMatrix, colPos, 0, true);
                        ppsSeq->findAndGetSint32(DCM_RowPositionInTotalImagePixelMatrix, rowPos, 0, true);
                        _positionToFrameIndex[(((rowPos - 1) / _tileHeight) * this->getSizeInTiles()[0]) + (((colPos - 1) / _tileWidth))] = i;
                    }
                }
            }
        }

        _isValid = true;
    }
}

WSIDicomInstance::~WSIDicomInstance()
{
    _isValid = false;
    _dataset = nullptr;
    _metaInfo = nullptr;
    _pixels = nullptr;
    delete _fileFormat;
    _fileFormat = nullptr;
    DJDecoderRegistration::cleanup();
}

std::string WSIDicomInstance::getUID(const std::string& UIDName) const { return this->_UIDs.at(UIDName); }

const std::map<std::string, std::string>& WSIDicomInstance::getUIDs() const { return _UIDs; }

std::vector<double> WSIDicomInstance::getPixelSpacing() const
{
    return { this->_pixelSpacingX, this->_pixelSpacingY };
}

std::vector<unsigned int> WSIDicomInstance::getSize() const
{
    return { _width, _height };
}

std::vector<unsigned short> WSIDicomInstance::getTileSize() const
{
    return { _tileWidth, _tileHeight };
}

std::vector<unsigned short> WSIDicomInstance::getSizeInTiles() const
{
    std::vector<unsigned short> sizeInTiles = { static_cast<unsigned short>(std::ceil(static_cast<float>(_width) / _tileWidth)),
                                                static_cast<unsigned short>(std::ceil(static_cast<float>(_height) / _tileHeight)) };
    return sizeInTiles;
}

WSIDicomInstance::DcmImageType WSIDicomInstance::getImageType() const
{
    return _imageType;
}

bool WSIDicomInstance::valid() const {
    return _isValid;
}

void* WSIDicomInstance::getFrame(const long long& x, const long long& y, const long long& z, const long long& op)
{
    Uint32 bufferSize = 0;
    _pixels->getUncompressedFrameSize(_dataset, bufferSize);
    unsigned char* buffer = new unsigned char[bufferSize];
    std::fill(buffer, buffer + bufferSize, 0); 
    int frameRow = y / _tileHeight;
    int frameColumn = x / _tileWidth;
    std::vector<unsigned short> sizeInTiles = this->getSizeInTiles();
    int frameOffset = this->_tiling == TilingType::Sparse ? _positionToFrameIndex[frameRow * sizeInTiles[0] + frameColumn]  : frameColumn + sizeInTiles[0] * frameRow;
    Uint32 fragmentNo = 0;
    OFString dcmColorModel = "";
    _pixels->getUncompressedFrame(_dataset, frameOffset, fragmentNo, buffer, bufferSize, dcmColorModel);
    return buffer;
}

