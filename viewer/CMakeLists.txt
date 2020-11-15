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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhGlobalAvgPoolLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhMaxPoolLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhPaddingLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhRes2BlockLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhResBlockLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhRouteLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhVariableOpLayer.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhPermuteLayer.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhReductionLayer.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhViewLayer.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhSoftMaxLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhUpSampleLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhSliceLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhYoloLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/layers/MsnhYoloOutLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/net/MsnhNetBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/net/MsnhNetwork.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/MsnhExString.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/MsnhExVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/MsnhTimeUtil.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/binary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/contrib/graphbuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/contrib/graphbuilderadapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/directives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/emit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/emitfromevents.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/emitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/emitterstate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/emitterutils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/exceptions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/exp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/node_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/nodebuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/nodeevents.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/null.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/ostream_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/parse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/regex_yaml.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/scanner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/scanscalar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/scantag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/scantoken.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/simplekey.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/singledocparser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/tag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/collectionstack.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/contrib/graphbuilderadapter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/directives.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/emitterstate.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/emitterutils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/exp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/indentation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/nodebuilder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/nodeevents.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/ptr_vector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/regex_yaml.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/regeximpl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/scanner.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/scanscalar.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/scantag.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/setting.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/singledocparser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/stream.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/streamcharsource.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/stringsource.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/tag.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/yaml_cpp/token.h
    )

add_executable(MsnhnetViewer ${SRCS})
target_include_directories(MsnhnetViewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(MsnhnetViewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(MsnhnetViewer Qt5::Widgets)
target_link_libraries(MsnhnetViewer Qt5::Svg)
target_link_libraries(MsnhnetViewer Qt5::PrintSupport)
target_link_libraries(MsnhnetViewer ${OMP_LIB})
install(TARGETS MsnhnetViewer
    RUNTIME DESTINATION bin)
