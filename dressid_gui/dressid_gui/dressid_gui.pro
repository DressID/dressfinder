#-------------------------------------------------
#
# Project created by QtCreator 2017-12-05T05:41:18
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = dressid_gui
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += /usr/local/include/opencv2
LIBS += -L/usr/local/include/opencv2
LIBS += -lopencv_core \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -lopencv_highgui \
        -lopencv_objdetect \
        -lopencv_ml \
        -lopencv_features2d \
        -lopencv_flann

SOURCES += \
        main.cpp \
        model_main.cpp \
    ANNtrainer.cpp \
    style.cpp \
    qdress.cpp

HEADERS += \
        model_main.h \
    ANNtrainer.hpp \
    qdress.h

FORMS += \
        model_main.ui
