/****************************************************************************
** Meta object code from reading C++ file 'ASAP_GUI_Window.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../ASAP_GUI_Window.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ASAP_GUI_Window.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.10.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ASAP__Worklist__GUI__ASAP_GUI_Window_t {
    QByteArrayData data[12];
    char stringdata0[168];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ASAP__Worklist__GUI__ASAP_GUI_Window_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ASAP__Worklist__GUI__ASAP_GUI_Window_t qt_meta_stringdata_ASAP__Worklist__GUI__ASAP_GUI_Window = {
    {
QT_MOC_LITERAL(0, 0, 36), // "ASAP::Worklist::GUI::ASAP_GUI..."
QT_MOC_LITERAL(1, 37, 16), // "OnWorklistClear_"
QT_MOC_LITERAL(2, 54, 0), // ""
QT_MOC_LITERAL(3, 55, 5), // "index"
QT_MOC_LITERAL(4, 61, 3), // "row"
QT_MOC_LITERAL(5, 65, 6), // "column"
QT_MOC_LITERAL(6, 72, 16), // "OnPatientsClear_"
QT_MOC_LITERAL(7, 89, 13), // "OnStudyClear_"
QT_MOC_LITERAL(8, 103, 17), // "OnWorklistSelect_"
QT_MOC_LITERAL(9, 121, 16), // "OnPatientSelect_"
QT_MOC_LITERAL(10, 138, 14), // "OnStudySelect_"
QT_MOC_LITERAL(11, 153, 14) // "OnImageSelect_"

    },
    "ASAP::Worklist::GUI::ASAP_GUI_Window\0"
    "OnWorklistClear_\0\0index\0row\0column\0"
    "OnPatientsClear_\0OnStudyClear_\0"
    "OnWorklistSelect_\0OnPatientSelect_\0"
    "OnStudySelect_\0OnImageSelect_"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ASAP__Worklist__GUI__ASAP_GUI_Window[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    3,   49,    2, 0x08 /* Private */,
       6,    3,   56,    2, 0x08 /* Private */,
       7,    3,   63,    2, 0x08 /* Private */,
       8,    1,   70,    2, 0x08 /* Private */,
       9,    1,   73,    2, 0x08 /* Private */,
      10,    1,   76,    2, 0x08 /* Private */,
      11,    1,   79,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::QModelIndex, QMetaType::Int, QMetaType::Int,    3,    4,    5,
    QMetaType::Void, QMetaType::QModelIndex, QMetaType::Int, QMetaType::Int,    3,    4,    5,
    QMetaType::Void, QMetaType::QModelIndex, QMetaType::Int, QMetaType::Int,    3,    4,    5,
    QMetaType::Void, QMetaType::QModelIndex,    3,
    QMetaType::Void, QMetaType::QModelIndex,    3,
    QMetaType::Void, QMetaType::QModelIndex,    3,
    QMetaType::Void, QMetaType::QModelIndex,    3,

       0        // eod
};

void ASAP::Worklist::GUI::ASAP_GUI_Window::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ASAP_GUI_Window *_t = static_cast<ASAP_GUI_Window *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->OnWorklistClear_((*reinterpret_cast< QModelIndex(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 1: _t->OnPatientsClear_((*reinterpret_cast< QModelIndex(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 2: _t->OnStudyClear_((*reinterpret_cast< QModelIndex(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 3: _t->OnWorklistSelect_((*reinterpret_cast< QModelIndex(*)>(_a[1]))); break;
        case 4: _t->OnPatientSelect_((*reinterpret_cast< QModelIndex(*)>(_a[1]))); break;
        case 5: _t->OnStudySelect_((*reinterpret_cast< QModelIndex(*)>(_a[1]))); break;
        case 6: _t->OnImageSelect_((*reinterpret_cast< QModelIndex(*)>(_a[1]))); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ASAP::Worklist::GUI::ASAP_GUI_Window::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_ASAP__Worklist__GUI__ASAP_GUI_Window.data,
      qt_meta_data_ASAP__Worklist__GUI__ASAP_GUI_Window,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *ASAP::Worklist::GUI::ASAP_GUI_Window::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ASAP::Worklist::GUI::ASAP_GUI_Window::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ASAP__Worklist__GUI__ASAP_GUI_Window.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int ASAP::Worklist::GUI::ASAP_GUI_Window::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
