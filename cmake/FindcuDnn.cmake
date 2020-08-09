

set(CUDNN_HINTS
${CUDA_ROOT}
$ENV{CUDA_ROOT}
$ENV{CUDA_TOOLKIT_ROOT_DIR}
)

set(CUDNN_PATHS
/usr
/usr/local
/usr/local/cuda
)

find_path(CUDNN_INCUDE_DIRS
    NAMES cudnn.h
    HINTS ${CUDNN_HINTS}
    PATH_SUFFIXES include inc include/x86_64 include/x64
    PATHS ${CUDNN_PATHS}
    DOC "cudnn include header cudnn.h"
)
mark_as_advanced(CUDNN_INCUDE_DIRS)

find_library(CUDNN_LIBRARIES
    NAMES cudnn
    HINTS ${CUDNN_HINTS}
    PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import lib/aarch64-linux-gpu
    PATHS ${CUDNN_PATHS}
    DOC "cudnn library"
)
mark_as_advanced(CUDNN_LIBRARIES)

# ===============================================
if(NOT CUDNN_INCUDE_DIRS)
    message(STATUS "Could NOT find 'cudnn.h', install CUDA/cuDnn or set CUDA_ROOT")
endif()

if(NOT CUDNN_LIBRARIES)
    message(STATUS "Could NOT find cuDnn library, install it or set CUDA_ROOT")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDnn DEFAULT_MSG CUDNN_INCUDE_DIRS CUDNN_LIBRARIES)

# =================================================