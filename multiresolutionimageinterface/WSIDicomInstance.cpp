#include "WSIDicomInstance.h"

#include <vector>

#include "dcmtk/config/osconfig.h"
#include "dcmtk/ofstd/ofcond.h"
#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcuid.h"
#include "dcmtk/dcmdata/dcfilefo.h"
#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmdata/dcxfer.h"
#include "dcmtk/dcmdata/dcmetinf.h"
#include "dcmtk/dcmdata/dcrlerp.h"
#include "dcmtk/dcmdata/dcrledrg.h"

// JPEG en/decoding
#include "dcmtk/dcmjpeg/djdecode.h"
#include "dcmtk/dcmjpeg/djencode.h"

const std::vector<std::string> WSIDicomInstance::SUPPORTED_TRANSFER_SYNTAX = { UID_JPEGProcess1TransferSyntax, UID_JPEGProcess2_4TransferSyntax, UID_JPEG2000LosslessOnlyTransferSyntax, UID_JPEG2000TransferSyntax };

WSIDicomInstance::WSIDicomInstance() :
    _fileFormat(nullptr), _dataset(nullptr), _metaInfo(nullptr), _image(nullptr), _isValid(false),
    _frameOffset(0), _numberOfFrames(0), _imageType(DcmImageType::InvalidImageType), _tiling(TilingType::Sparse)
{
}

WSIDicomInstance::~WSIDicomInstance()
{
    _isValid = false;
    _dataset = nullptr;
    _metaInfo = nullptr;
    _image = nullptr;
    delete _fileFormat;
    _fileFormat = nullptr;
}

bool WSIDicomInstance::initialize(DcmFileFormat* fileFormat) {

    OFString msSOPClassUID;
    std::vector<DcmTagKey> requiredTags = { DCM_StudyInstanceUID, DCM_SeriesInstanceUID, DCM_FrameOfReferenceUID, DCM_Rows, DCM_Columns,
                                            DCM_SamplesPerPixel, DCM_PhotometricInterpretation, DCM_ImageType, DCM_TotalPixelMatrixColumns,
                                            DCM_TotalPixelMatrixRows, DCM_NumberOfFrames, DCM_SharedFunctionalGroupsSequence, DCM_OpticalPathSequence };
    this->_metaInfo = fileFormat->getMetaInfo();
    this->_dataset = fileFormat->getDataset();
    this->_metaInfo->findAndGetOFString(DCM_MediaStorageSOPClassUID, msSOPClassUID);
    if (msSOPClassUID == UID_VLWholeSlideMicroscopyImageStorage) {
        DcmXfer transferSyntax = _dataset->getOriginalXfer();
        if (std::find(this->SUPPORTED_TRANSFER_SYNTAX.begin(), this->SUPPORTED_TRANSFER_SYNTAX.end(), transferSyntax.getXferID()) == this->SUPPORTED_TRANSFER_SYNTAX.end()) {
            return _isValid;
        }
        DcmElement* tagValue;
        for (auto tag : requiredTags) {
            if (this->_dataset->findAndGetElement(tag, tagValue).bad()) {
                return _isValid;
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
        } else {
            _imageType = DcmImageType::InvalidImageType;
            return _isValid;
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
                return _isValid;
            }
        }
        if (this->_dataset->findAndGetUint32(DCM_NumberOfFrames, this->_numberOfFrames).bad()) {
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
        double pixelSpacingX = 0 , pixelSpacingY = 0, sliceSpacing = 0;
        if (this->_dataset->findAndGetSequenceItem(DCM_SharedFunctionalGroupsSequence, sfgSeq).good()) {
            DcmItem* pmSeq;
            if (sfgSeq->findAndGetSequenceItem(DCM_PixelMeasuresSequence, pmSeq).good()) {
                pmSeq->findAndGetFloat64(DCM_PixelSpacing, pixelSpacingX, 0);
                pmSeq->findAndGetFloat64(DCM_PixelSpacing, pixelSpacingY, 1);
                pmSeq->findAndGetFloat64(DCM_SpacingBetweenSlices, sliceSpacing);
                if (pixelSpacingX == 0 || pixelSpacingY == 0) {
                    return _isValid;
                }
            }
        }



        _isValid = true;
    }
	return _isValid;
}
