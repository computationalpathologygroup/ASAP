#-------------------------------------------------
#
# Project created by QtCreator 2014-01-09T14:15:24
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets


TARGET = PathologyWorkstation


SOURCES += main.cpp\
        pathologyworkstation.cpp\
        PathologyViewer.cpp

HEADERS  += pathologyworkstation.h\
            PathologyViewer.h

FORMS    += pathologyworkstation.ui

win32: LIBS += -L"$$PWD/../../../../build (x64_VC11)/bin/Debug/" -lmultiresolutionimageinterface_d

DEPENDPATH += $$PWD/../../

INCLUDEPATH += $$PWD/../../../\
               $$PWD/../../\
               $$PWD/../\
               $$PWD/
