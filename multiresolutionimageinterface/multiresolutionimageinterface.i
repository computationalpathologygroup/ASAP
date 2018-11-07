%begin %{
#if defined(_DEBUG) && defined(SWIG_PYTHON_INTERPRETER_NO_DEBUG)
/* https://github.com/swig/swig/issues/325 */
# include <basetsd.h>
# include <assert.h>
# include <ctype.h>
# include <errno.h>
# include <io.h>
# include <math.h>
# include <sal.h>
# include <stdarg.h>
# include <stddef.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <sys/stat.h>
# include <time.h>
# include <wchar.h>
#endif
%}

%module multiresolutionimageinterface

%{
#define SWIG_FILE_WITH_INIT
#include "MultiResolutionImageReader.h"
#include "MultiResolutionImageWriter.h"
#include "MultiResolutionImage.h"
#include "TIFFImage.h"
#include "multiresolutionimageinterface_export.h"
#include "../config/ASAPMacros.h"
#include "../core/Point.h"
#include "../core/ProgressMonitor.h"
#include "../core/CmdLineProgressMonitor.h"
#include "../annotation/AnnotationBase.h"
#include "../annotation/Annotation.h"
#include "../annotation/AnnotationGroup.h"
#include "../annotation/AnnotationList.h"
#include "../annotation/AnnotationService.h"
#include "../annotation/AnnotationToMask.h"
#include "../annotation/Repository.h"
#include "../annotation/XmlRepository.h"
#include "../annotation/NDPARepository.h"
#include "../annotation/ImageScopeRepository.h"
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"
%include "std_shared_ptr.i"

%shared_ptr(ImageSource)
%shared_ptr(MultiResolutionImage)
%shared_ptr(OpenSlideImage)
%shared_ptr(VSIImage)
%shared_ptr(TIFFImage)
%shared_ptr(LIFImage)
%shared_ptr(Annotation)
%shared_ptr(AnnotationBase)
%shared_ptr(AnnotationGroup)
%shared_ptr(AnnotationList)
%shared_ptr(Repository)
%shared_ptr(XmlRepository)
%shared_ptr(NDPARepository)
%shared_ptr(ImageScopeRepository)

namespace std {
  %template(vector_int) vector<int>;
  %template(vector_uint) vector<unsigned int>;
  %template(vector_float) vector<float>;
  %template(vector_double) vector<double>;
  %template(vector_annotation) vector<shared_ptr<Annotation> >;
  %template(vector_annotation_group) vector<shared_ptr<AnnotationGroup> >;
  %template(vector_unsigned_long_long) vector<unsigned long long>;
  %template(vector_long_long) vector<long long>;
  %template(vector_string) vector<string>;
  %template(vector_point) vector<Point>;
  %template(map_int_string) map<int, string>;
  %template(map_string_int) map<string, int>;
}
%include "numpy.i"

%init %{
import_array();
%}

#ifdef SWIG
#define MULTIRESOLUTIONIMAGEINTERFACE_EXPORT
#define CORE_EXPORT
#define ANNOTATION_EXPORT
#endif

%ignore swap(ImageSource& first, ImageSource& second);

%immutable ASAP_VERSION_STRING;
%include "../config/ASAPMacros.h"

%include "../core/Point.h"
%include "../core/PathologyEnums.h"
%include "../core/ImageSource.h"
%include "../core/CmdLineProgressMonitor.h"
%include "../annotation/AnnotationBase.h"
%include "../annotation/Annotation.h"
%include "../annotation/AnnotationGroup.h"
%include "../annotation/AnnotationList.h"
%include "../annotation/AnnotationService.h"
%include "../annotation/AnnotationToMask.h"
%include "../annotation/Repository.h"
%include "../annotation/XmlRepository.h"
%include "../annotation/NDPARepository.h"
%include "../annotation/ImageScopeRepository.h"

%numpy_typemaps(void, NPY_NOTYPE, int)
%include "MultiResolutionImage.h";
%include "TIFFImage.h";
%extend MultiResolutionImage {
     void close() { 
		self->~MultiResolutionImage();
	}
};
%extend MultiResolutionImage {
     PyObject* getUCharPatch(const long long& startX, const long long& startY, const unsigned long long& width, 
						     const unsigned long long& height, const unsigned int& level) { 
		unsigned int nrSamples = self->getSamplesPerPixel();
        npy_intp dimsDesc[3];
		dimsDesc[0] = height;
		dimsDesc[1] = width;
		dimsDesc[2] = nrSamples;
		PyObject* patch = PyArray_SimpleNew(3, dimsDesc, NPY_UBYTE);
		unsigned char* tmp = new unsigned char[height*width*nrSamples];
		self->getRawRegion(startX, startY, width, height, level, tmp);
		unsigned char* array_data = (unsigned char*)PyArray_DATA((PyArrayObject*)patch);
		std::copy(tmp, tmp + height*width*nrSamples, array_data);
		delete[] tmp;
		return patch;
	}
};
%extend MultiResolutionImage {
     PyObject* getUInt16Patch(const long long& startX, const long long& startY, const unsigned long long& width, 
						     const unsigned long long& height, const unsigned int& level) { 
		unsigned int nrSamples = self->getSamplesPerPixel();
        npy_intp dimsDesc[3];
		dimsDesc[0] = height;
		dimsDesc[1] = width;
		dimsDesc[2] = nrSamples;
		PyObject* patch = PyArray_SimpleNew(3, dimsDesc, NPY_UINT16);
		unsigned short* tmp = new unsigned short[height*width*nrSamples];
		self->getRawRegion(startX, startY, width, height, level, tmp);
		unsigned short* array_data = (unsigned short*)PyArray_DATA((PyArrayObject*)patch);
		std::copy(tmp, tmp + height*width*nrSamples, array_data);
		delete[] tmp;
		return patch;
	}
};
%extend MultiResolutionImage {
     PyObject* getUInt32Patch(const long long& startX, const long long& startY, const unsigned long long& width, 
						     const unsigned long long& height, const unsigned int& level) { 
		unsigned int nrSamples = self->getSamplesPerPixel();
        npy_intp dimsDesc[3];
		dimsDesc[0] = height;
		dimsDesc[1] = width;
		dimsDesc[2] = nrSamples;
		PyObject* patch = PyArray_SimpleNew(3, dimsDesc, NPY_UINT32);
		unsigned int* tmp = new unsigned int[height*width*nrSamples];
		self->getRawRegion(startX, startY, width, height, level, tmp);
		unsigned int* array_data = (unsigned int*)PyArray_DATA((PyArrayObject*)patch);
		std::copy(tmp, tmp + height*width*nrSamples, array_data);
		delete[] tmp;
		return patch;
	}
};
%extend MultiResolutionImage {
     PyObject* getFloatPatch(const long long& startX, const long long& startY, const unsigned long long& width, 
						     const unsigned long long& height, const unsigned int& level) { 
		unsigned int nrSamples = self->getSamplesPerPixel();
        npy_intp dimsDesc[3];
		dimsDesc[0] = height;
		dimsDesc[1] = width;
		dimsDesc[2] = nrSamples;
		PyObject* patch = PyArray_SimpleNew(3, dimsDesc, NPY_FLOAT);
		float* tmp = new float[height*width*nrSamples];
		self->getRawRegion(startX, startY, width, height, level, tmp);
		float* array_data = (float*)PyArray_DATA((PyArrayObject*)patch);
		std::copy(tmp, tmp + height*width*nrSamples, array_data);
		delete[] tmp;
		return patch;
	}
};
%extend TIFFImage {
     PyObject* getEncodedTile(const long long& startX, const long long& startY, const unsigned int& level) { 
		long long encoded_tile_size = self->getEncodedTileSize(startX, startY, level);
		if (encoded_tile_size > 0) {
			npy_intp dimsDesc[1];
			dimsDesc[0] = encoded_tile_size;
			PyObject* patch = PyArray_SimpleNew(1, dimsDesc, NPY_UBYTE);
			unsigned char* tmp = self->readEncodedDataFromImage(startX, startY, level);
			unsigned char* array_data = (unsigned char*)PyArray_DATA((PyArrayObject*)patch);
			std::copy(tmp, tmp + encoded_tile_size, array_data);
			delete[] tmp;
			return patch;
		} else {
			return NULL;
		}
	}
};

%inline %{
  std::shared_ptr<TIFFImage> MultiResolutionImageToTIFFImage(std::shared_ptr<MultiResolutionImage> base) {
    return std::dynamic_pointer_cast<TIFFImage>(base);
  }
%}

%include "MultiResolutionImageReader.h"

%apply (void* IN_ARRAY1_UNKNOWN_SIZE) {(void* data)};
%include "MultiResolutionImageWriter.h"