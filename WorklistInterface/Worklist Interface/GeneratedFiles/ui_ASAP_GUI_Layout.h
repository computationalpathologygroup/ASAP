/********************************************************************************
** Form generated from reading UI file 'ASAP_GUI_Layout.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ASAP_GUI_LAYOUT_H
#define UI_ASAP_GUI_LAYOUT_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTableView>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QTreeView>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ASAP_GUI_Layout
{
public:
    QWidget *centralWidget;
    QTreeView *WorklistView;
    QTableView *PatientView;
    QTableView *StudyView;
    QListView *ImageView;
    QLabel *label_worklists;
    QLabel *label_patients;
    QLabel *label_studies;
    QLabel *label_images;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *ASAP_GUI_Layout)
    {
        if (ASAP_GUI_Layout->objectName().isEmpty())
            ASAP_GUI_Layout->setObjectName(QStringLiteral("ASAP_GUI_Layout"));
        ASAP_GUI_Layout->resize(1000, 850);
        centralWidget = new QWidget(ASAP_GUI_Layout);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        WorklistView = new QTreeView(centralWidget);
        WorklistView->setObjectName(QStringLiteral("WorklistView"));
        WorklistView->setGeometry(QRect(10, 40, 256, 741));
        WorklistView->setRootIsDecorated(true);
        WorklistView->setHeaderHidden(true);
        PatientView = new QTableView(centralWidget);
        PatientView->setObjectName(QStringLiteral("PatientView"));
        PatientView->setGeometry(QRect(270, 40, 721, 190));
        PatientView->horizontalHeader()->setCascadingSectionResizes(true);
        PatientView->horizontalHeader()->setProperty("showSortIndicator", QVariant(false));
        PatientView->horizontalHeader()->setStretchLastSection(true);
        PatientView->verticalHeader()->setVisible(false);
        PatientView->verticalHeader()->setDefaultSectionSize(33);
        PatientView->verticalHeader()->setMinimumSectionSize(20);
        StudyView = new QTableView(centralWidget);
        StudyView->setObjectName(QStringLiteral("StudyView"));
        StudyView->setGeometry(QRect(270, 270, 721, 190));
        StudyView->verticalHeader()->setVisible(false);
        ImageView = new QListView(centralWidget);
        ImageView->setObjectName(QStringLiteral("ImageView"));
        ImageView->setGeometry(QRect(270, 500, 721, 281));
        ImageView->setIconSize(QSize(200, 200));
        ImageView->setResizeMode(QListView::Fixed);
        ImageView->setViewMode(QListView::IconMode);
        label_worklists = new QLabel(centralWidget);
        label_worklists->setObjectName(QStringLiteral("label_worklists"));
        label_worklists->setGeometry(QRect(20, 10, 121, 21));
        label_patients = new QLabel(centralWidget);
        label_patients->setObjectName(QStringLiteral("label_patients"));
        label_patients->setGeometry(QRect(270, 10, 71, 21));
        label_studies = new QLabel(centralWidget);
        label_studies->setObjectName(QStringLiteral("label_studies"));
        label_studies->setGeometry(QRect(270, 240, 101, 21));
        label_images = new QLabel(centralWidget);
        label_images->setObjectName(QStringLiteral("label_images"));
        label_images->setGeometry(QRect(270, 470, 81, 21));
        ASAP_GUI_Layout->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ASAP_GUI_Layout);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1000, 24));
        ASAP_GUI_Layout->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ASAP_GUI_Layout);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        ASAP_GUI_Layout->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(ASAP_GUI_Layout);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        ASAP_GUI_Layout->setStatusBar(statusBar);

        retranslateUi(ASAP_GUI_Layout);

        QMetaObject::connectSlotsByName(ASAP_GUI_Layout);
    } // setupUi

    void retranslateUi(QMainWindow *ASAP_GUI_Layout)
    {
        ASAP_GUI_Layout->setWindowTitle(QApplication::translate("ASAP_GUI_Layout", "Worklistinterface", nullptr));
        label_worklists->setText(QApplication::translate("ASAP_GUI_Layout", "Worklists", nullptr));
        label_patients->setText(QApplication::translate("ASAP_GUI_Layout", "Patients", nullptr));
        label_studies->setText(QApplication::translate("ASAP_GUI_Layout", "Studies", nullptr));
        label_images->setText(QApplication::translate("ASAP_GUI_Layout", "Images", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ASAP_GUI_Layout: public Ui_ASAP_GUI_Layout {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ASAP_GUI_LAYOUT_H
