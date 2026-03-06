QT += core gui openglwidgets svg

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

SOURCES += \
    GLWidget.cpp \
    MarchingCubes.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    GLWidget.h \
    MarchingCubes.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

RESOURCES += \
    resources.qrc

DISTFILES += \
    shader.frag \
    shader.vert

# OpenCV
INCLUDEPATH += C:/Users/weiti/Desktop/opencv/build/include

# Release mode
CONFIG(release, debug|release) {
    LIBS += -LC:/Users/weiti/Desktop/opencv/build/x64/vc16/lib -lopencv_world4120
}

# Debug mode
CONFIG(debug, debug|release) {
    LIBS += -LC:/Users/weiti/Desktop/opencv/build/x64/vc16/lib -lopencv_world4120d
}

