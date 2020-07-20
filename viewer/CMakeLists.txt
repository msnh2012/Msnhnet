find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5Svg REQUIRED)
find_package(Qt5PrintSupport REQUIRED)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
# Instruct CMake to create code from Qt designer ui files
set(CMAKE_AUTOUIC ON)
# Instruct CMake to run Qt ressource
set(CMAKE_AUTORCC ON)


set(SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/MsnhBlas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/MsnhGemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/io/MsnhIO.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/io/MsnhParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhActivationLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhActivations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhActivationsAvx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhActivationsNeon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhAddBlockLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhBaseLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhBatchNormLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhConcatBlockLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhConnectedLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhConvolutionalLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhCropLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhDeConvolutionalLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhEmptyLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhLocalAvgPoolLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhMaxPoolLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhPaddingLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhRes2BlockLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhResBlockLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhRouteLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhSoftMaxLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhUpSampleLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhYolov3Layer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhYolov3OutLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/net/MsnhNetBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/net/MsnhNetwork.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/MsnhExString.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/MsnhExVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerAttribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerColorTabel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerLink.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerMemberFrm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerNodeCreator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerNodeSelect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerScene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerThemeManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MsnhViewerView.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mainwindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ressources/resources.qrc
    )

add_executable(MsnhnetViewer ${SRCS})
target_include_directories(MsnhnetViewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(MsnhnetViewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(MsnhnetViewer Qt5::Widgets)
target_link_libraries(MsnhnetViewer Qt5::Svg)
target_link_libraries(MsnhnetViewer Qt5::PrintSupport)
target_link_libraries(MsnhnetViewer yaml-cpp)
install(TARGETS MsnhnetViewer
    RUNTIME DESTINATION bin)
