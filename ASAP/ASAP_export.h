
#ifndef ASAP_EXPORT_H
#define ASAP_EXPORT_H

#ifdef ASAP_STATIC_DEFINE
#  define ASAP_EXPORT
#  define ASAP_NO_EXPORT
#else
#  ifndef ASAP_EXPORT
#  ifdef _WIN32
#    ifdef ASAP_EXPORTS
        /* We are building this library */
#      define ASAP_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define ASAP_EXPORT __declspec(dllimport)
#    endif
#  else
#    define ASAP_EXPORT
# endif
#  endif

#  ifndef ASAP_NO_EXPORT
#    define ASAP_NO_EXPORT 
#  endif
#endif

#ifndef ASAP_DEPRECATED
#  define ASAP_DEPRECATED __declspec(deprecated)
#endif

#ifndef ASAP_DEPRECATED_EXPORT
#  define ASAP_DEPRECATED_EXPORT ASAP_EXPORT ASAP_DEPRECATED
#endif

#ifndef ASAP_DEPRECATED_NO_EXPORT
#  define ASAP_DEPRECATED_NO_EXPORT ASAP_NO_EXPORT ASAP_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define ASAP_NO_DEPRECATED
#endif

#endif
