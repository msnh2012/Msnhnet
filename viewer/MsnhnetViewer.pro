QT       += core gui printsupport svg

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += $${PWD}/../include
INCLUDEPATH += D:/libs/yaml/include
SOURCES += \
    MsnhViewerAttribute.cpp \
    MsnhViewerColorTabel.cpp \
    MsnhViewerLink.cpp \
    MsnhViewerMemberFrm.cpp \
    MsnhViewerNode.cpp \
    MsnhViewerNodeCreator.cpp \
    MsnhViewerNodeSelect.cpp \
    MsnhViewerScene.cpp \
    MsnhViewerThemeManager.cpp \
    MsnhViewerView.cpp \
    ../src/3rdparty/yaml_cpp/binary.cpp\
    ../src/3rdparty/yaml_cpp/contrib/graphbuilder.cpp\
    ../src/3rdparty/yaml_cpp/contrib/graphbuilderadapter.cpp\
    ../src/3rdparty/yaml_cpp/convert.cpp\
    ../src/3rdparty/yaml_cpp/directives.cpp\
    ../src/3rdparty/yaml_cpp/emit.cpp\
    ../src/3rdparty/yaml_cpp/emitfromevents.cpp\
    ../src/3rdparty/yaml_cpp/emitter.cpp\
    ../src/3rdparty/yaml_cpp/emitterstate.cpp\
    ../src/3rdparty/yaml_cpp/emitterutils.cpp\
    ../src/3rdparty/yaml_cpp/exceptions.cpp\
    ../src/3rdparty/yaml_cpp/exp.cpp\
    ../src/3rdparty/yaml_cpp/memory.cpp\
    ../src/3rdparty/yaml_cpp/node.cpp\
    ../src/3rdparty/yaml_cpp/node_data.cpp\
    ../src/3rdparty/yaml_cpp/nodebuilder.cpp\
    ../src/3rdparty/yaml_cpp/nodeevents.cpp\
    ../src/3rdparty/yaml_cpp/null.cpp\
    ../src/3rdparty/yaml_cpp/ostream_wrapper.cpp\
    ../src/3rdparty/yaml_cpp/parse.cpp\
    ../src/3rdparty/yaml_cpp/parser.cpp\
    ../src/3rdparty/yaml_cpp/regex_yaml.cpp\
    ../src/3rdparty/yaml_cpp/scanner.cpp\
    ../src/3rdparty/yaml_cpp/scanscalar.cpp\
    ../src/3rdparty/yaml_cpp/scantag.cpp\
    ../src/3rdparty/yaml_cpp/scantoken.cpp\
    ../src/3rdparty/yaml_cpp/simplekey.cpp\
    ../src/3rdparty/yaml_cpp/singledocparser.cpp\
    ../src/3rdparty/yaml_cpp/stream.cpp\
    ../src/3rdparty/yaml_cpp/tag.cpp\
    ../src/3rdparty/yaml_cpp/collectionstack.h\
    ../src/3rdparty/yaml_cpp/contrib/graphbuilderadapter.h\
    ../src/3rdparty/yaml_cpp/directives.h\
    ../src/3rdparty/yaml_cpp/emitterstate.h\
    ../src/3rdparty/yaml_cpp/emitterutils.h\
    ../src/3rdparty/yaml_cpp/exp.h\
    ../src/3rdparty/yaml_cpp/indentation.h\
    ../src/3rdparty/yaml_cpp/nodebuilder.h\
    ../src/3rdparty/yaml_cpp/nodeevents.h\
    ../src/3rdparty/yaml_cpp/ptr_vector.h\
    ../src/3rdparty/yaml_cpp/regex_yaml.h\
    ../src/3rdparty/yaml_cpp/regeximpl.h\
    ../src/3rdparty/yaml_cpp/scanner.h\
    ../src/3rdparty/yaml_cpp/scanscalar.h\
    ../src/3rdparty/yaml_cpp/scantag.h\
    ../src/3rdparty/yaml_cpp/setting.h\
    ../src/3rdparty/yaml_cpp/singledocparser.h\
    ../src/3rdparty/yaml_cpp/stream.h\
    ../src/3rdparty/yaml_cpp/streamcharsource.h\
    ../src/3rdparty/yaml_cpp/stringsource.h\
    ../src/3rdparty/yaml_cpp/tag.h\
    ../src/3rdparty/yaml_cpp/token.h\
    ../src/layers/MsnhActivationsAvx.cpp \
    ../src/layers/MsnhActivationsNeon.cpp \
    ../src/core/MsnhBlas.cpp \
    ../src/core/MsnhGemm.cpp \
    ../src/io/MsnhIO.cpp \
    ../src/io/MsnhParser.cpp \
    ../src/layers/MsnhActivationLayer.cpp \
    ../src/layers/MsnhActivations.cpp \
    ../src/layers/MsnhAddBlockLayer.cpp \
    ../src/layers/MsnhBaseLayer.cpp \
    ../src/layers/MsnhBatchNormLayer.cpp \
    ../src/layers/MsnhConcatBlockLayer.cpp \
    ../src/layers/MsnhConnectedLayer.cpp \
    ../src/layers/MsnhConvolutionalLayer.cpp \
    ../src/layers/MsnhCropLayer.cpp \
    ../src/layers/MsnhDeConvolutionalLayer.cpp \
    ../src/layers/MsnhEmptyLayer.cpp \
    ../src/layers/MsnhLocalAvgPoolLayer.cpp \
    ../src/layers/MsnhGlobalAvgPoolLayer.cpp \
    ../src/layers/MsnhMaxPoolLayer.cpp \
    ../src/layers/MsnhPaddingLayer.cpp \
    ../src/layers/MsnhRes2BlockLayer.cpp \
    ../src/layers/MsnhPermuteLayer.cpp \
    ../src/layers/MsnhViewLayer.cpp \
    ../src/layers/MsnhResBlockLayer.cpp \
    ../src/layers/MsnhReductionLayer.cpp \
    ../src/layers/MsnhVariableOpLayer.cpp\
    ../src/layers/MsnhRouteLayer.cpp \
    ../src/layers/MsnhVariableOpLayer.cpp \
    ../src/layers/MsnhSliceLayer.cpp \
    ../src/layers/MsnhSoftMaxLayer.cpp \
    ../src/layers/MsnhUpSampleLayer.cpp \
    ../src/layers/MsnhYoloLayer.cpp \
    ../src/layers/MsnhYoloOutLayer.cpp \
    ../src/net/MsnhNetBuilder.cpp \
    ../src/net/MsnhNetwork.cpp \
    ../src/utils/MsnhExString.cpp \
    ../src/utils/MsnhExVector.cpp \
    ../src/utils/MsnhTimeUtil.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    MsnhViewerAttribute.h \
    MsnhViewerColorTabel.h \
    MsnhViewerLink.h \
    MsnhViewerMemberFrm.h \
    MsnhViewerNode.h \
    MsnhViewerNodeCfg.h \
    MsnhViewerNodeCreator.h \
    MsnhViewerNodeSelect.h \
    MsnhViewerScene.h \
    MsnhViewerThemeManager.h \
    MsnhViewerTypes.h \
    MsnhViewerView.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    ressources/resources.qrc

DEFINES +=  USE_OMP
DEFINES +=  USE_X86
