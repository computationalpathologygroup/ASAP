/********************************************************************************
** Form generated from reading UI file 'WorklistWindowLayout.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WORKLISTWINDOWLAYOUT_H
#define UI_WORKLISTWINDOWLAYOUT_H

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

class Ui_WorklistWindowLayout
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

    void setupUi(QMainWindow *WorklistWindowLayout)
    {
        if (WorklistWindowLayout->objectName().isEmpty())
            WorklistWindowLayout->setObjectName(QStringLiteral("WorklistWindowLayout"));
        WorklistWindowLayout->resize(1000, 850);
        centralWidget = new QWidget(WorklistWindowLayout);
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
        WorklistWindowLayout->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(WorklistWindowLayout);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1000, 24));
        WorklistWindowLayout->setMenuBar(menuBar);
        mainToolBar = new QToolBar(WorklistWindowLayout);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        WorklistWindowLayout->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(WorklistWindowLayout);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        WorklistWindowLayout->setStatusBar(statusBar);

        retranslateUi(WorklistWindowLayout);

        QMetaObject::connectSlotsByName(WorklistWindowLayout);
    } // setupUi

    void retranslateUi(QMainWindow *WorklistWindowLayout)
    {
        WorklistWindowLayout->setWindowTitle(QApplication::translate("WorklistWindowLayout", "Worklistinterface", nullptr));
        label_worklists->setText(QApplication::translate("WorklistWindowLayout", "Worklists", nullptr));
        label_patients->setText(QApplication::translate("WorklistWindowLayout", "Patients", nullptr));
        label_studies->setText(QApplication::translate("WorklistWindowLayout", "Studies", nullptr));
        label_images->setText(QApplication::translate("WorklistWindowLayout", "Images", nullptr));
    } // retranslateUi

};

namespace Ui {
    class WorklistWindowLayout: public Ui_WorklistWindowLayout {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WORKLISTWINDOWLAYOUT_H
