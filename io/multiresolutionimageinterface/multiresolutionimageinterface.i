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
#include "../../core/Point.h"
#include "../../core/ProgressMonitor.h"
#include "../../core/CmdLineProgressMonitor.h"
#include "../../annotation/Annotation.h"
#include "../../annotation/AnnotationGroup.h"
#include "../../annotation/AnnotationList.h"
#include "../../annotation/AnnotationService.h"
#include "../../annotation/AnnotationToMask.h"
#include "../../annotation/Repository.h"
#include "../../annotation/XmlRepository.h"
#include "../../annotation/NDPARepository.h"
#include "../../annotation/ImageScopeRepository.h"
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"

namespace std {
  %template(vector_int) vector<int>;
  %template(vector_uint) vector<unsigned int>;
  %template(vector_float) vector<float>;
  %template(vector_double) vector<double>;
  %template(vector_annotation) vector<Annotation*>;
  %template(vector_annotation_group) vector<AnnotationGroup*>;
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
#define EXPORT_MULTIRESOLUTIONIMAGEINTERFACE
#define EXPORT_CORE
#define EXPORT_PATHOLOGYANNOTATION
#endif

%ignore swap(ImageSource& first, ImageSource& second);

%include "../../core/Point.h"
%include "../../core/PathologyEnums.h"
%include "../../core/ImageSource.h"
%include "../../core/CmdLineProgressMonitor.h"

%include "../../annotation/Annotation.h"
%include "../../annotation/AnnotationGroup.h"
%include "../../annotation/AnnotationList.h"
%include "../../annotation/AnnotationService.h"
%include "../../annotation/AnnotationToMask.h"
%include "../../annotation/Repository.h"
%include "../../annotation/XmlRepository.h"
%include "../../annotation/NDPARepository.h"
%include "../../annotation/ImageScopeRepository.h"

%numpy_typemaps(void, NPY_NOTYPE, int)
%include "MultiResolutionImage.h";
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
		self->getRawRegion<unsigned char>(startX, startY, width, height, level, tmp);
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
		self->getRawRegion<unsigned short>(startX, startY, width, height, level, tmp);
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
		self->getRawRegion<unsigned int>(startX, startY, width, height, level, tmp);
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
		self->getRawRegion<float>(startX, startY, width, height, level, tmp);
		float* array_data = (float*)PyArray_DATA((PyArrayObject*)patch);
		std::copy(tmp, tmp + height*width*nrSamples, array_data);
		delete[] tmp;
		return patch;
	}
};
%include "MultiResolutionImageReader.h"

%apply (void* IN_ARRAY1_UNKNOWN_SIZE) {(void* data)};
%include "MultiResolutionImageWriter.h"