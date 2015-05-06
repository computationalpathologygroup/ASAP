/***************************************************************-*-c++-*-

  @COPYRIGHT@

  $Id: diag_config.h,v 1.27 2007/06/27 15:23:12 yulia Exp $

*************************************************************************/

#ifndef __PATHOLOGYCONFIG_H__
#define __PATHOLOGYCONFIG_H__

#ifdef __WIN32__
  #ifndef WIN32
    #define WIN32
  #endif
#endif
#ifdef WIN32
 #ifndef _WIN32_
  #define _WIN32_
 #endif
 #ifndef __WIN32__
  #define __WIN32__
 #endif
#endif

// Configuration file, handles support for various compilers

// under windows we currently support Borland and MSVC
// the following is for both compilers
#if defined(__BORLANDC__) || defined(_MSC_VER)

  // Defines to handle import/export in DLLs
  #  if defined(BUILD_CORE)
  #    define EXPORT_CORE __declspec(dllexport)
  #  else
  #    define EXPORT_CORE __declspec(dllimport)
  #  endif
	
  #  ifdef BUILD_MULTIRESOLUTIONIMAGEINTERFACE
  #    define EXPORT_MULTIRESOLUTIONIMAGEINTERFACE __declspec(dllexport)
  #  else
  #    define EXPORT_MULTIRESOLUTIONIMAGEINTERFACE __declspec(dllimport)
  #  endif

  #  ifdef BUILD_PATHOLOGYANNOTATION
  #    define EXPORT_PATHOLOGYANNOTATION __declspec(dllexport)
  #  else
  #    define EXPORT_PATHOLOGYANNOTATION __declspec(dllimport)
  #  endif

  #  ifdef BUILD_BASICFILTERS
  #    define EXPORT_BASICFILTERS __declspec(dllexport)
  #  else
  #    define EXPORT_BASICFILTERS __declspec(dllimport)
  #  endif

  #  ifdef BUILD_SLIDESTANDARDIZATION
  #    define EXPORT_SLIDESTANDARDIZATION __declspec(dllexport)
  #  else
  #    define EXPORT_SLIDESTANDARDIZATION __declspec(dllimport)
  #  endif

  #  ifdef BUILD_SUPERPIXELCLASSIFICATION
  #    define EXPORT_SUPERPIXELCLASSIFICATION __declspec(dllexport)
  #  else
  #    define EXPORT_SUPERPIXELCLASSIFICATION __declspec(dllimport)
  #  endif
  
  # ifdef BUILD_NUCLEISEGMENTATION
  #    define EXPORT_NUCLEISEGMENTATION __declspec(dllexport)
  #  else
  #    define EXPORT_NUCLEISEGMENTATION __declspec(dllimport)
  #  endif  

  #  ifdef BUILD_ANNOTATIONTOOLS
  #    define EXPORT_ANNOTATIONTOOLS __declspec(dllexport)
  #  else
  #    define EXPORT_ANNOTATIONTOOLS __declspec(dllimport)
  #  endif

  # ifdef BUILD_PATHOLOGYWORKSTATION
  #    define EXPORT_PATHOLOGYWORKSTATION __declspec(dllexport)
  #  else
  #    define EXPORT_PATHOLOGYWORKSTATION __declspec(dllimport)
  #  endif

  #  ifdef BUILD_GRAPHTOOLS
  #    define EXPORT_GRAPHTOOLS __declspec(dllexport)
  #  else
  #    define EXPORT_GRAPHTOOLS __declspec(dllimport)
  #  endif

  #  ifdef BUILD_OBJECTFEATURE
  #    define EXPORT_OBJECTFEATURE __declspec(dllexport)
  #  else
  #    define EXPORT_OBJECTFEATURE __declspec(dllimport)
  #  endif

  #  ifdef BUILD_BASICFILTERS
  #    define EXPORT_BASICFILTERS __declspec(dllexport)
  #  else
  #    define EXPORT_BASICFILTERS __declspec(dllimport)
  #  endif

  # ifdef BUILD_VISUALIZATIONPLUGIN
  #    define EXPORT_VISUALIZATIONPLUGIN __declspec(dllexport)
  #  else
  #    define EXPORT_VISUALIZATIONPLUGIN __declspec(dllimport)
  #  endif

#else
  #  define EXPORT_CORE
  #  define EXPORT_MULTIRESOLUTIONIMAGEINTERFACE
  #  define EXPORT_PATHOLOGYANNOTATION
  #  define EXPORT_SUPERPIXELCLASSIFICATION
  #  define EXPORT_SLIDESTANDARDIZATION
  #  define EXPORT_ANNOTATIONTOOLS
  #  define EXPORT_NUCLEISEGMENTATION
  #  define EXPORT_PATHOLOGYWORKSTATION
  #  define EXPORT_GRAPHTOOLS
  #  define EXPORT_OBJECTFEATURE
  #  define EXPORT_BASICFILTERS
#endif

#ifdef __WIN32__
  // stuff specific for MSVC
  #if (defined _MSC_VER) || (defined __GNUC__)
	  #pragma warning( disable : 4251 )
  #endif
#endif

#endif