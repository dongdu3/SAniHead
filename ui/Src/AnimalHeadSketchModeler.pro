#-------------------------------------------------
#
# Project created by QtCreator 2018-10-30T22:36:56
#
#-------------------------------------------------
QT += core gui widgets opengl

TARGET = AnimalHeadSketchModeler
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
CONFIG += no_keywords c++11

FORMS += \
    Widgets/mainwindow.ui

HEADERS += \
    Widgets/mainwindow.h \
    MeshRenderer/ArcBall.h \
    MeshRenderer/Opengl.h \
    MeshRenderer/ViewObj.h \
    Widgets/sketchwidget.h \
    Widgets/meshwidget.h \
    MeshGenerator/tensorflowmodel.h \
    MeshRenderer/suggestivecontour.h \
    MeshProcessor/LineIntersection/LineSegmentIntersection.h \
    MeshProcessor/Deformation/LaplacianDeformation.h \
    MeshProcessor/LineIntersection/raytriangleintersection.h \
    MeshProcessor/BaseProcessor.h \
    MeshProcessor/DijkstraShortestPath.h

SOURCES += \
    main.cpp \
    Widgets/mainwindow.cpp \
    MeshRenderer/ArcBall.cpp \
    MeshRenderer/Opengl.cpp \
    MeshRenderer/ViewObj.cpp \
    Widgets/meshwidget.cpp \
    Widgets/sketchwidget.cpp \
    MeshRenderer/suggestivecontour.cpp \
    MeshProcessor/LineIntersection/LineSegmentIntersection.cpp \
    MeshProcessor/Deformation/LaplacianDeformation.cpp \
    MeshProcessor/BaseProcessor.cpp \
    MeshProcessor/DijkstraShortestPath.cpp

# add library inlude path and lib file
INCLUDEPATH += $$PWD/../Library/
INCLUDEPATH += $$PWD/../Library/glew \
    $$PWD/../Library/trimesh2/include \
#    $$PWD/../Library/libigl/include

LIBS += -L$$PWD/../Library/glew/lib.Linux64/ -lGLEW  \
    -L$$PWD/../Library/trimesh2/lib.Linux64/ -ltrimesh -lgluit \
    -L/usr/lib/x86_64-linux-gnu/ -lX11 -lGLU -lglut \


# for python3
INCLUDEPATH += /usr/include/python3.6
INCLUDEPATH += /usr/lib/python3.6/config-3.6m-x86_64-linux-gnu
DEPENDPATH += /usr/lib/python3.6/config-3.6m-x86_64-linux-gnu

LIBS += -L/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/ -lpython3.6m
unix:!macx: PRE_TARGETDEPS += /usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6m.a

LIBS += -L/usr/local/cuda-10.0/lib64/ -lcublas

LIBS += -fopenmp
