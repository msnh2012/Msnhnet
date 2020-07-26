
cmake_minimum_required(VERSION 3.10)

project(unet
    LANGUAGES CXX C
    VERSION 1.0)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Msnhnet REQUIRED)
find_package(OpenMP REQUIRED)


add_executable(${PROJECT_NAME} unet.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${Msnhnet_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} yaml-cpp Msnhnet)
