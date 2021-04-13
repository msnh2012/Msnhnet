cmake_minimum_required(VERSION 3.15) # cuda

# jetson rum cmake in terminal
project(Msnhnet
    LANGUAGES CXX C
    VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)

message(STATUS "OS is ${CMAKE_SYSTEM}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(ENABLE_OMP           "Use omp to speed up"       ON ) 
option(OMP_MAX_THREAD       "Omp will use max threads"  ON )
option(ENABLE_OPEN_BLAS     "Use open blas"             OFF)
option(BUILD_SHARED_LIBS    "Build shared lib"          ON )
option(BUILD_EXAMPLES       "Build examples"            ON )
option(BUILD_USE_OPENCV     "Build with Opencv"         OFF)
option(BUILD_VIEWER         "Build MsnhnetViewer"       OFF)
option(BUILD_USE_GPU        "Use Gpu"                   OFF )
option(BUILD_USE_CUDNN      "Use cuDnn"                 OFF )
option(BUILD_USE_MSNHCV_GUI "Use msnhCV GUI"            OFF)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_Release} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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

if(BUILD_USE_MSNHCV_GUI MATCHES ON)
    add_definitions(-DUSE_MSNHCV_GUI)
    set(USE_MSNHCV_GUI_MACRO "#define USE_MSNHCV_GUI\n") #===============
    message(STATUS "Use msnhCV GUI")
    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)
    # CMAKE_PREFIX_PATH "your path"
    find_package(GLEW REQUIRED)
    set(MSNHCV_GUI_SRC
        src/3rdparty/imgui/ImCurveEdit.cpp
        src/3rdparty/imgui/ImGradient.cpp
        src/3rdparty/imgui/ImGuizmo.cpp
        src/3rdparty/imgui/ImSequencer.cpp
        src/3rdparty/imgui/imgui.cpp
        src/3rdparty/imgui/imgui_demo.cpp
        src/3rdparty/imgui/imgui_draw.cpp
        src/3rdparty/imgui/imgui_impl_glfw.cpp
        src/3rdparty/imgui/imgui_impl_opengl3.cpp
        src/3rdparty/imgui/imgui_widgets.cpp
        src/3rdparty/imgui/imnodes.cpp
        src/3rdparty/imgui/implot.cpp
        src/3rdparty/imgui/implot_demo.cpp
        src/3rdparty/imgui/implot_items.cpp
        )
else()
    set(USE_MSNHCV_GUI_MACRO "")
    set(MSNHCV_GUI_SRC "")
endif()

set(SRCS
    src/core/MsnhBlas.cpp
    src/core/MsnhGemm.cpp
    src/core/MsnhSimd.cpp
    src/core/MsnhMemoryManager.cpp
    src/io/MsnhIO.cpp
    src/io/MsnhParser.cpp
    src/layers/MsnhActivationLayer.cpp
    src/layers/MsnhActivations.cpp
    src/layers/MsnhActivationsAvx.cpp
    src/layers/MsnhActivationsNeon.cpp
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
    src/layers/MsnhGlobalAvgPoolLayer.cpp
    src/layers/MsnhMaxPoolLayer.cpp
    src/layers/MsnhPaddingLayer.cpp
    src/layers/MsnhPermuteLayer.cpp
    src/layers/MsnhPixelShuffleLayer.cpp
    src/layers/MsnhViewLayer.cpp
    src/layers/MsnhReductionLayer.cpp
    src/layers/MsnhRes2BlockLayer.cpp
    src/layers/MsnhResBlockLayer.cpp
    src/layers/MsnhRouteLayer.cpp
    src/layers/MsnhSoftMaxLayer.cpp
    src/layers/MsnhUpSampleLayer.cpp
    src/layers/MsnhYoloLayer.cpp
    src/layers/MsnhYoloOutLayer.cpp
    src/layers/MsnhVariableOpLayer.cpp 
    src/layers/MsnhSliceLayer.cpp 
    src/net/MsnhNetBuilder.cpp
    src/net/MsnhNetwork.cpp
    src/utils/MsnhExString.cpp
    src/utils/MsnhExVector.cpp
    src/utils/MsnhMathUtils.cpp
    src/utils/MsnhOpencvUtil.cpp
    src/utils/MsnhTimeUtil.cpp
    src/c_api/MsnhnetLib.cpp
    src/layers/x86/MsnhConvolution3x3LayerX86.cpp
    src/layers/arm/armv7a/MsnhConvolution3x3s1.cpp
    src/layers/arm/armv7a/MsnhConvolution3x3s2.cpp
    src/layers/arm/armv7a/MsnhConvolution3x3s1Winograd.cpp
    src/layers/arm/armv7a/MsnhConvolutionSgemm.cpp
    src/layers/arm/armv7a/MsnhBatchNorm.cpp
    src/layers/arm/armv7a/MsnhPadding.cpp
    src/layers/arm/armv7a/MsnhActivation.cpp
    src/3rdparty/yaml_cpp/binary.cpp
    src/3rdparty/yaml_cpp/contrib/graphbuilder.cpp
    src/3rdparty/yaml_cpp/contrib/graphbuilderadapter.cpp
    src/3rdparty/yaml_cpp/convert.cpp
    src/3rdparty/yaml_cpp/directives.cpp
    src/3rdparty/yaml_cpp/emit.cpp
    src/3rdparty/yaml_cpp/emitfromevents.cpp
    src/3rdparty/yaml_cpp/emitter.cpp
    src/3rdparty/yaml_cpp/emitterstate.cpp
    src/3rdparty/yaml_cpp/emitterutils.cpp
    src/3rdparty/yaml_cpp/exceptions.cpp
    src/3rdparty/yaml_cpp/exp.cpp
    src/3rdparty/yaml_cpp/memory.cpp
    src/3rdparty/yaml_cpp/node.cpp
    src/3rdparty/yaml_cpp/node_data.cpp
    src/3rdparty/yaml_cpp/nodebuilder.cpp
    src/3rdparty/yaml_cpp/nodeevents.cpp
    src/3rdparty/yaml_cpp/null.cpp
    src/3rdparty/yaml_cpp/ostream_wrapper.cpp
    src/3rdparty/yaml_cpp/parse.cpp
    src/3rdparty/yaml_cpp/parser.cpp
    src/3rdparty/yaml_cpp/regex_yaml.cpp
    src/3rdparty/yaml_cpp/scanner.cpp
    src/3rdparty/yaml_cpp/scanscalar.cpp
    src/3rdparty/yaml_cpp/scantag.cpp
    src/3rdparty/yaml_cpp/scantoken.cpp
    src/3rdparty/yaml_cpp/simplekey.cpp
    src/3rdparty/yaml_cpp/singledocparser.cpp
    src/3rdparty/yaml_cpp/stream.cpp
    src/3rdparty/yaml_cpp/tag.cpp
    src/3rdparty/yaml_cpp/collectionstack.h
    src/3rdparty/yaml_cpp/contrib/graphbuilderadapter.h
    src/3rdparty/yaml_cpp/directives.h
    src/3rdparty/yaml_cpp/emitterstate.h
    src/3rdparty/yaml_cpp/emitterutils.h
    src/3rdparty/yaml_cpp/exp.h
    src/3rdparty/yaml_cpp/indentation.h
    src/3rdparty/yaml_cpp/nodebuilder.h
    src/3rdparty/yaml_cpp/nodeevents.h
    src/3rdparty/yaml_cpp/ptr_vector.h
    src/3rdparty/yaml_cpp/regex_yaml.h
    src/3rdparty/yaml_cpp/regeximpl.h
    src/3rdparty/yaml_cpp/scanner.h
    src/3rdparty/yaml_cpp/scanscalar.h
    src/3rdparty/yaml_cpp/scantag.h
    src/3rdparty/yaml_cpp/setting.h
    src/3rdparty/yaml_cpp/singledocparser.h
    src/3rdparty/yaml_cpp/stream.h
    src/3rdparty/yaml_cpp/streamcharsource.h
    src/3rdparty/yaml_cpp/stringsource.h
    src/3rdparty/yaml_cpp/tag.h
    src/3rdparty/yaml_cpp/token.h
    src/cv/MsnhCVDraw.cpp 
    src/cv/MsnhCVFilters.cpp 
    src/cv/MsnhCVFont.cpp
    src/cv/MsnhCVGeometry.cpp
    src/cv/MsnhCVGui.cpp
    src/cv/MsnhCVMat.cpp 
    src/cv/MsnhCVMatOp.cpp 
    src/cv/MsnhCVVideo.cpp
    src/robot/MsnhJoint.cpp
    src/robot/MsnhFrame.cpp
    src/robot/MsnhChain.cpp
    src/robot/MsnhSegment.cpp
    src/robot/MsnhChainFK.cpp
    src/utils/MsnhCVUtil.cpp
    src/robot/MsnhSpatialMath.cpp
    )

# config GPU
if(BUILD_USE_GPU MATCHES ON)
    add_definitions(-DUSE_GPU)#===============

    enable_language(CUDA) 

    set(USE_GPU_MACRO "#define USE_GPU\n")

    set(CXX_CUDA_SRCS 
        src/config/MsnhnetCuda.cpp
    )

    set(CUDA_SRCS
        src/config/cuda/MsnhnetCuda.cu
        src/core/cuda/MsnhBlasGPU.cu
        src/core/cuda/MsnhGemmGPU.cu
        src/layers/cuda/MsnhActivationsGPU.cu
        src/layers/cuda/MsnhConvolutionalLayerGPU.cu
        src/layers/cuda/MsnhMaxPoolLayerGPU.cu
        src/layers/cuda/MsnhLocalAvgPoolLayerGPU.cu
        src/layers/cuda/MsnhGlobalAvgPoolLayerGPU.cu
        src/layers/cuda/MsnhPaddingLayerGPU.cu
        src/layers/cuda/MsnhPermuteLayerGPU.cu
        src/layers/cuda/MsnhPixelShuffleLayerGPU.cu
        src/layers/cuda/MsnhYoloLayerGPU.cu
        src/layers/cuda/MsnhYoloOutLayerGPU.cu
        src/layers/cuda/MsnhSliceLayerGPU.cu
        )
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
    find_package(cuBlas REQUIRED)

    if(BUILD_USE_CUDNN MATCHES ON)
        add_definitions(-DUSE_CUDNN)#===============
        find_package(cuDnn REQUIRED)
        set(USE_CUDNN_MACRO "#define USE_CUDNN\n")
    else()
        set(USE_CUDNN_MACRO )
    endif()

else()
    set(USE_GPU_MACRO )
    set(USE_CUDNN_MACRO )
    set(CUDA_SRCS )
    set(CXX_CUDA_SRCS )
endif()

set(CUDA_INC_DIR    )

# X86 config
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(x86)|(X86)|(amd64)|(AMD64)")
    add_definitions(-DUSE_X86)#===============
    set(USE_X86_MACRO "#define USE_X86\n")
    message(STATUS "Use ${CMAKE_SYSTEM_PROCESSOR} arch")

   # debug mode with _d on windows with msvc
    if(MSVC)
        set(CMAKE_DEBUG_POSTFIX "_d")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
    else()
	set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -march=native -O3 -msse4.1 -mssse3 -msse3 -msse2 -msse -mavx -mavx2")
	set(CUDA_INC_DIR   /usr/local/cuda/include)	
    endif()

else()
    set(USE_X86_MACRO "")
endif()

message(${CMAKE_SYSTEM_PROCESSOR})

# ARM config
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(armv7l)")
    add_definitions(-DUSE_ARM) #===============
    set(USE_ARM_MACRO "#define USE_ARM\n")
    set(CUDA_INC_DIR   /usr/local/cuda/include)

    message(STATUS "Use arm arch")

    option(ENABLE_NEON "Use neon" OFF)

    if(ENABLE_NEON MATCHES ON)
        add_definitions(-DUSE_NEON)
        set(USE_NEON_MACRO "#define USE_NEON\n")

        if (CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7l)") # aarch64 neon is default
            set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -mfpu=neon-vfpv4")
        endif()
        message(STATUS "Use neon")
    else()
        set(USE_NEON_MACRO "")
    endif()
else()
    set(USE_ARM_MACRO "")
endif()

# find 3rdparty libs
set(USE_OPENCV_MACRO "")

if(BUILD_USE_OPENCV MATCHES ON)
find_package(OpenCV REQUIRED)
add_definitions(-DUSE_OPENCV)
message(STATUS "OpenCV Dir ${OpenCV_DIR}")
message(STATUS "OpenCV include dir ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs ${OpenCV_LIBS}")
set(USE_OPENCV_MACRO "#define USE_OPENCV\n")
endif()
# find_package(yaml-cpp REQUIRED)
# message(STATUS "Found yaml in ${yaml-cpp_DIR}")
# message(STATUS "Yaml version ${yaml-cpp_VERSION}")
# message(STATUS "yaml include dir ${YAML_CPP_INCLUDE_DIR}")
# message(STATUS "yaml libs ${YAML_CPP_LIBRARIES}")

if(BUILD_SHARED_LIBS MATCHES ON)
    add_library(${PROJECT_NAME} SHARED ${SRCS} ${CUDA_SRCS})     #generate libs
    set(USE_SHARED_MSNHNET_MACRO "#define USE_SHARED_MSNHNET\n") #===============
    if(UNIX OR APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
    message("shared--------------------")
else()
    add_library(${PROJECT_NAME} STATIC ${SRCS} ${CUDA_SRCS})     #generate libs
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
    ${CUDA_INC_DIR}
    )

#source
target_sources(${PROJECT_NAME}
    PRIVATE
    ${SRCS}
    ${CXX_CUDA_SRCS}
    ${MSNHCV_GUI_SRC}
    )

#openblas
if(ENABLE_OPEN_BLAS MATCHES ON)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenBLAS_LIBRARIES})
endif()

# dshow
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PUBLIC gdi32 ole32 strmiids oleaut32 user32 shlwapi)
endif()

# opengl
if(BUILD_USE_MSNHCV_GUI MATCHES ON)
    target_include_directories(${PROJECT_NAME} PUBLIC ${GLEW_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${GLEW_LIBRARIES} glfw ${OPENGL_LIBRARIES})
endif()

# opencv
if(BUILD_USE_OPENCV MATCHES ON)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
endif()

if(BUILD_USE_GPU MATCHES ON)
    if(BUILD_USE_CUDNN MATCHES ON)
        target_link_libraries(${PROJECT_NAME} PUBLIC ${OMP_LIB} PRIVATE ${CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES})
    else()
        target_link_libraries(${PROJECT_NAME} PUBLIC ${OMP_LIB} PRIVATE ${CUBLAS_LIBRARIES})
    endif()
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OMP_LIB})
endif()

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
file(WRITE MsnhnetTmp/MsnhnetMacro.h "#ifndef MSNHNETMACRO_H\n#define MSNHNETMACRO_H\n${USE_OMP_MACRO}${OMP_THREAD_MACRO}${USE_OPEN_BLAS_MACRO}${USE_X86_MACRO}${USE_ARM_MACRO}${USE_NEON_MACRO}${USE_OPENCV_MACRO}${USE_SHARED_MSNHNET_MACRO}${USE_GPU_MACRO}${USE_CUDNN_MACRO}${USE_MSNHCV_GUI_MACRO}#endif")

# Install macro head file
install(FILES MsnhnetTmp/MsnhnetMacro.h
        DESTINATION include/${PROJECT_NAME}/config
)

