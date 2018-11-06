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

%module wholeslidefilters

%{
#define SWIG_FILE_WITH_INIT
#include "config/ASAPMacros.h"
#include "wholeslidefilters_export.h"
#include "DistanceTransformWholeSlideFilter.h"
#include "ConnectedComponentsWholeSlideFilter.h"
#include "NucleiDetectionWholeSlideFilter.h"
#include "LabelStatisticsWholeSlideFilter.h"
#include "ThresholdWholeSlideFilter.h"
#include "ArithmeticWholeSlideFilter.h"
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"

namespace std {
  %template(vector_int) vector<int>;
  %template(vector_uint) vector<unsigned int>;
  %template(vector_float) vector<float>;
  %template(vector_double) vector<double>;
  %template(vector_vector_float) vector<vector< float> >;
  %template(vector_unsigned_long_long) vector<unsigned long long>;
  %template(vector_long_long) vector<long long>;
  %template(vector_string) vector<string>;
}

#ifdef SWIG
#define WHOLESLIDEFILTERS_EXPORT
#define CORE_EXPORT
#endif

%immutable ASAP_VERSION_STRING;
%include "../../config/ASAPMacros.h"

%include "DistanceTransformWholeSlideFilter.h"
%include "ConnectedComponentsWholeSlideFilter.h"
%include "LabelStatisticsWholeSlideFilter.h"
%include "ThresholdWholeSlideFilter.h"
%include "ArithmeticWholeSlideFilter.h"
%include "NucleiDetectionWholeSlideFilter.h"