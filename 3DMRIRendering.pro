QT += core gui openglwidgets svg concurrent

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# CUDA Configuration
CUDA_PATH = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1
# GTX 1650 is compute capability 7.5, so compile for sm_75.
CUDA_GENCODE = -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75
NVCC = \"$$CUDA_PATH/bin/nvcc.exe\"

# Include CUDA headers (without quotes to avoid path splitting in Makefile)
INCLUDEPATH += $$CUDA_PATH/include

# CUDA libraries
LIBS += -L\"$$CUDA_PATH/lib/x64\" \
        -lcuda \
        -lcudart \
        -lcurand

# CUDA extra compiler configuration
CUDA_SOURCES = MarchingCubes.cu

cuda.input = CUDA_SOURCES
cuda.dependency_type = TYPE_C
cuda.variable_out = OBJECTS
cuda.name = NVCC ${QMAKE_FILE_IN}

CONFIG(debug, debug|release) {
    cuda.output = debug/${QMAKE_FILE_BASE}.obj
    cuda.commands = $$NVCC \
                    -std=c++17 \
                    -c \
                    -o \"${QMAKE_FILE_OUT}\" \
                    \"${QMAKE_FILE_IN}\" \
                    $$CUDA_GENCODE \
                    -I\"$$PWD\" \
                    -I\"$$CUDA_PATH/include\" \
                    -I\"$$[QT_INSTALL_HEADERS]\" \
                    -I\"$$[QT_INSTALL_HEADERS]/QtCore\" \
                    -I\"$$[QT_INSTALL_HEADERS]/QtGui\" \
                    -I\"$$[QT_INSTALL_HEADERS]/QtOpenGL\" \
                    -I\"C:/Users/weiti/Desktop/opencv/build/include\" \
                    --expt-relaxed-constexpr \
                    --compiler-options /permissive-,/Zc:__cplusplus,/EHsc,/W3,/nologo,/Od,/Zi,/MDd
} else {
    cuda.output = release/${QMAKE_FILE_BASE}.obj
    cuda.commands = $$NVCC \
                    -std=c++17 \
                    -c \
                    -o \"${QMAKE_FILE_OUT}\" \
                    \"${QMAKE_FILE_IN}\" \
                    $$CUDA_GENCODE \
                    -I\"$$PWD\" \
                    -I\"$$CUDA_PATH/include\" \
                    -I\"$$[QT_INSTALL_HEADERS]\" \
                    -I\"$$[QT_INSTALL_HEADERS]/QtCore\" \
                    -I\"$$[QT_INSTALL_HEADERS]/QtGui\" \
                    -I\"$$[QT_INSTALL_HEADERS]/QtOpenGL\" \
                    -I\"C:/Users/weiti/Desktop/opencv/build/include\" \
                    --expt-relaxed-constexpr \
                    --compiler-options /permissive-,/Zc:__cplusplus,/EHsc,/W3,/nologo,/O2,/MD
}

QMAKE_EXTRA_COMPILERS += cuda

SOURCES += \
    GLWidget.cpp \
    ImageLoader.cpp \
    ImageProcessor.cpp \
    LoginDialog.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    GLWidget.h \
    ImageLoader.h \
    ImageProcessor.h \
    LoginDialog.h \
    mainwindow.h \
    MarchingCubes.h

FORMS += \
    mainwindow.ui \
    LoginDialog.ui

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