cmake_minimum_required(VERSION 3.10) # cuda

project(Msnhnet
    LANGUAGES CXX C
    VERSION 1.0)

message(STATUS "OS is ${CMAKE_SYSTEM}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(ENABLE_OMP           "Use omp to speed up"       ON ) 
option(OMP_MAX_THREAD       "Omp will use max threads"  ON )
option(ENABLE_OPEN_BLAS     "Use open blas"             OFF)
option(BUILD_SHARED_LIBS    "Build shared lib"          ON )
option(BUILD_EXAMPLES       "Build examples"            ON )
option(BUILD_VIEWER         "Build MsnhnetViewer"       ON )


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_Release} -O3")

#use omp
if(ENABLE_OMP MATCHES ON)
    add_definitions(-DUSE_OMP)
    set(USE_OMP_MACRO "#define USE_OMP\n") #===============
    set(OMP_LIB OpenMP::OpenMP_CXX)
    message(STATUS "Use OMP")
    #find open mp
    find_package(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        message(STATUS "Found Openmp")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message(FATAL_ERROR  "Openmp not found!")
    endif()
else()
    set(USE_OMP_MACRO "")
    set(OMP_LIB )
endif()

# You can set thread num for omp
if((ENABLE_OMP MATCHES ON) AND ( OMP_MAX_THREAD MATCHES OFF) )
    message(WARNING "Please check OMP_THREAD num as OMP_MAX_THREAD is not be checked ")
    set(num 7)
    add_definitions(-DOMP_THREAD=${num})
    set(OMP_THREAD_MACRO "#define OMP_THREAD=${num}\n") #===============
else()
    set(OMP_THREAD_MACRO "")
endif()

if(ENABLE_OPEN_BLAS MATCHES ON)
    add_definitions(-DUSE_OPEN_BLAS)
    set(USE_OPEN_BLAS_MACRO "#define USE_OPEN_BLAS\n") #===============
    message(STATUS "Use open blas")
    find_package(OpenBLAS REQUIRED)
    if(OpenBLAS_FOUND)
        message(STATUS "Found openblas")
    else()
        message(FATAL_ERROR "Not found openblas")
    endif()
else()
    set(USE_OPEN_BLAS_MACRO "")
endif()

set(SRCS
    src/core/MsnhBlas.cpp
    src/core/MsnhGemm.cpp
    src/io/MsnhIO.cpp
    src/io/MsnhParser.cpp
    src/layers/MsnhActivationLayer.cpp
    src/layers/MsnhActivations.cpp
    src/layers/MsnhAddBlockLayer.cpp
    src/layers/MsnhBaseLayer.cpp
    src/layers/MsnhBatchNormLayer.cpp
    src/layers/MsnhConcatBlockLayer.cpp
    src/layers/MsnhConnectedLayer.cpp
    src/layers/MsnhConvolutionalLayer.cpp
    src/layers/MsnhCropLayer.cpp
    src/layers/MsnhDeConvolutionalLayer.cpp
    src/layers/MsnhEmptyLayer.cpp
    src/layers/MsnhLocalAvgPoolLayer.cpp
    src/layers/MsnhMaxPoolLayer.cpp
    src/layers/MsnhPaddingLayer.cpp
    src/layers/MsnhRes2BlockLayer.cpp
    src/layers/MsnhResBlockLayer.cpp
    src/layers/MsnhRouteLayer.cpp
    src/layers/MsnhSoftMaxLayer.cpp
    src/layers/MsnhUpSampleLayer.cpp
    src/layers/MsnhYolov3Layer.cpp
    src/layers/MsnhYolov3OutLayer.cpp
    src/net/MsnhNetBuilder.cpp
    src/net/MsnhNetwork.cpp
    src/utils/MsnhExString.cpp
    src/utils/MsnhExVector.cpp
    src/utils/MsnhMathUtils.cpp
    src/utils/MsnhOpencvUtil.cpp
    )

# X86 config
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    add_definitions(-DUSE_X86)#===============
    set(USE_X86_MACRO "#define USE_X86\n")
    message(STATUS "Use ${CMAKE_SYSTEM_PROCESSOR} arch")

   # debug mode with _d on windows with msvc
    if(MSVC)
        set(CMAKE_DEBUG_POSTFIX "_d")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    else()
        SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -msse4.1 -mssse3 -msse3 -msse2 -msse -mavx -mavx2")
    endif()
else()
    set(USE_X86_MACRO "")
endif()


# ARM config
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(aarch32)|(armv7l)")
    add_definitions(-DUSE_ARM) #===============
    set(USE_ARM_MACRO "#define USE_ARM\n")

    message(STATUS "Use arm arch")

   option(ENABLE_NEON "Use neon" ON)
    if(ENABLE_NEON MATCHES ON)
        add_definitions(-DUSE_NEON)
        set(USE_NEON_MACRO "#define USE_NEON\n")
        set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -mfpu=neon")
        message(STATUS "Use neon")
    else()
        set(USE_NEON_MACRO "")
    endif()
else()
    set(USE_ARM_MACRO "")
endif()

# find 3rdparty libs
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Dir ${OpenCV_DIR}")
message(STATUS "OpenCV include dir ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs ${OpenCV_LIBS}")
find_package(yaml-cpp REQUIRED)
message(STATUS "Found yaml in ${yaml-cpp_DIR}")
message(STATUS "Yaml version ${yaml-cpp_VERSION}")
message(STATUS "yaml include dir ${YAML_CPP_INCLUDE_DIR}")
message(STATUS "yaml libs ${YAML_CPP_LIBRARIES}")

if(BUILD_SHARED_LIBS MATCHES ON)
    add_library(${PROJECT_NAME} SHARED ${SRCS})   #generate libs
    set(USE_SHARED_MSNHNET_MACRO "#define USE_SHARED_MSNHNET\n") #===============
    message("shared--------------------")
else()
    add_library(${PROJECT_NAME} STATIC ${SRCS})   #generate libs
    set(USE_SHARED_MSNHNET_MACRO "")
    message("static--------------------")
endif()

#add_library(${PROJECT_NAME}::Msnhnet ALIAS Msnhnet)

if(BUILD_SHARED_LIBS MATCHES ON)
    # add definitions for target
    target_compile_definitions(${PROJECT_NAME}
                               PRIVATE EXPORT_MSNHNET) # private definitions will not affect other target eg. examples
else()
    # add definitions for target
    target_compile_definitions(${PROJECT_NAME}
                               PRIVATE EXPORT_MSNHNET EXPORT_MSNHNET_STATIC) # private definitions will not affect other target eg. examples
endif()

#source headers
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    )

#source
target_sources(${PROJECT_NAME}
    PRIVATE
    ${SRCS}
    )

#openblas
if(ENABLE_OPEN_BLAS MATCHES ON)
    target_link_libraries(${PROJECT_NAME} ${OpenBLAS_LIBRARIES})
endif()

#link 3rdparty libs
target_link_libraries(${PROJECT_NAME} PUBLIC ${OMP_LIB} ${OpenCV_LIBS} yaml-cpp)

if(BUILD_EXAMPLES MATCHES ON)
    add_subdirectory(examples)
endif()

if(BUILD_VIEWER MATCHES ON)
    add_subdirectory(viewer)
endif()

# install
install(TARGETS ${PROJECT_NAME}
    EXPORT  ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib  # dynamic
    ARCHIVE DESTINATION lib  # static
    RUNTIME DESTINATION bin  # exe
    INCLUDES DESTINATION include  #header
    )
# version
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion  # Compatible with older version
    )

#install targets
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
    )

#install headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")

#install images
install(DIRECTORY ${PROJECT_SOURCE_DIR}/images
        DESTINATION .)

#install labels
install(DIRECTORY ${PROJECT_SOURCE_DIR}/labels
        DESTINATION .)

#install cmake config
configure_file(${PROJECT_NAME}Config.cmake.in ${PROJECT_NAME}Config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME}
    )

# Create cfg macro head file
file(WRITE MsnhnetTmp/MsnhnetMacro.h "#ifndef MSNHNETMACRO_H\n#define MSNHNETMACRO_H\n${USE_OMP_MACRO}${OMP_THREAD_MACRO}${USE_OPEN_BLAS_MACRO}${USE_X86_MACRO}${USE_ARM_MACRO}${USE_NEON_MACRO}${USE_SHARED_MSNHNET_MACRO}#endif")

# Install macro head file
install(FILES MsnhnetTmp/MsnhnetMacro.h
        DESTINATION include/${PROJECT_NAME}/config
)

