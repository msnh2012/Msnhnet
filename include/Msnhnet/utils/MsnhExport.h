#ifndef MSNHEXPORT_H
#define MSNHEXPORT_H

#ifdef EXPORT_MSNHNET_STATIC
    #define MsnhNet_EXPORT
    #define MsnhNet_IMPORT
#else
    #ifdef WIN32
        #define MsnhNet_EXPORT __declspec(dllexport)
        #define MsnhNet_IMPORT __declspec(dllimport)
    #elif __GNUC__ >= 4 || __clang__
        #define MsnhNet_EXPORT __attribute__((visibility("default")))
        #define MsnhNet_IMPORT __attribute__((visibility("default")))
    #else
        #define MsnhNet_EXPORT
        #define MsnhNet_IMPORT
    #endif
#endif

#ifdef EXPORT_MSNHNET
    #define MsnhNet_API MsnhNet_EXPORT
#else
    #ifdef USE_SHARED_MSNHNET
        #define MsnhNet_API MsnhNet_IMPORT
    #else
        #define MsnhNet_API
    #endif
#endif

#ifdef WIN32
#pragma warning( disable: 4251 )
#endif

#endif 

