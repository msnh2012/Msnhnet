
cmake_minimum_required(VERSION 3.10)

project(unet_gpu
    LANGUAGES CXX C CUDA
    VERSION 1.0)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Msnhnet REQUIRED)
find_package(OpenMP REQUIRED)


add_executable(unet_gpu unet_gpu.cpp)
add_executable(unet_gpu_only_cpu unet_gpu_only_cpu.cpp)

target_include_directories(unet_gpu PUBLIC ${Msnhnet_INCLUDE_DIR})
target_include_directories(unet_gpu_only_cpu PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(unet_gpu PUBLIC ${OpenCV_LIBS} yaml-cpp Msnhnet)
target_link_libraries(unet_gpu_only_cpu PUBLIC ${OpenCV_LIBS} yaml-cpp Msnhnet)
