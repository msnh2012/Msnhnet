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
    ../src/layers/MsnhResBlockLayer.cpp \
    ../src/layers/MsnhReductionLayer.cpp \
    ../src/layers/MsnhVariableOpLayer.cpp\
    ../src/layers/MsnhRouteLayer.cpp \
    ../src/layers/MsnhVariableOpLayer.cpp \
    ../src/layers/MsnhSoftMaxLayer.cpp \
    ../src/layers/MsnhUpSampleLayer.cpp \
    ../src/layers/MsnhYolov3Layer.cpp \
    ../src/layers/MsnhYolov3OutLayer.cpp \
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

CONFIG(debug,debug|release){                                                #debug模式
LIBS         += D:/libs/yaml/lib/libyaml-cppmdd.lib
}

CONFIG(release,debug|release){                                              #release模式
LIBS         +=  D:/libs/yaml/lib/libyaml-cppmd.lib
}

DISTFILES +=
