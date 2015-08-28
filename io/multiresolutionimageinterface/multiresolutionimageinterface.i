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
%include "../../core/pathologyEnums.h"
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
%numpy_typemaps(float             , NPY_FLOAT    , long long)
%numpy_typemaps(unsigned char     , NPY_UBYTE    , long long)

%apply (unsigned char*& INPLACE_ARRAY1, long long DIM1) {(unsigned char*& data, long long size)};
%apply (float*& INPLACE_ARRAY1, long long DIM1) {(float*& data, long long size)};
%include "MultiResolutionImage.h";
%extend MultiResolutionImage {
     void getUCharPatch(const long long& startX, const long long& startY, const unsigned long long& width, 
						const unsigned long long& height, const unsigned int& level, unsigned char*& data, long long size) { 
		if (width*height*self->getSamplesPerPixel() == size) {
		    unsigned char* tmp = new unsigned char[size];
			self->getRawRegion<unsigned char>(startX, startY, width, height, level, tmp);
			std::copy(tmp, tmp + size, data);
			delete[] tmp;
	    } else {
		    PyErr_Format(PyExc_IndexError, "array not the same size as requested image");
		}
	}
};
%extend MultiResolutionImage {
     void getFloatPatch(const long long& startX, const long long& startY, const unsigned long long& width, 
						const unsigned long long& height, const unsigned int& level, float*& data, long long size) { 
		if (width*height*self->getSamplesPerPixel() == size) {
			float* tmp = new float[size];
			self->getRawRegion<float>(startX, startY, width, height, level, tmp);
			std::copy(tmp, tmp + size, data);
			delete[] tmp;
	    } else {
		    PyErr_Format(PyExc_IndexError, "array not the same size as requested image");
		}
	}
};
%include "MultiResolutionImageReader.h"

%apply (void* IN_ARRAY1_UNKNOWN_SIZE) {(void* data)};
%include "MultiResolutionImageWriter.h"