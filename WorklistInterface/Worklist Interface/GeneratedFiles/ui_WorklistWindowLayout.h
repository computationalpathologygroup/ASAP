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
#include <QtWidgets/QGridLayout>
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
    QGridLayout *gridLayout;
    QLabel *label_worklists;
    QLabel *label_patients;
    QTreeView *WorklistView;
    QTableView *PatientView;
    QLabel *label_studies;
    QTableView *StudyView;
    QLabel *label_images;
    QListView *ImageView;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *WorklistWindowLayout)
    {
        if (WorklistWindowLayout->objectName().isEmpty())
            WorklistWindowLayout->setObjectName(QStringLiteral("WorklistWindowLayout"));
        WorklistWindowLayout->resize(1113, 850);
        centralWidget = new QWidget(WorklistWindowLayout);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label_worklists = new QLabel(centralWidget);
        label_worklists->setObjectName(QStringLiteral("label_worklists"));

        gridLayout->addWidget(label_worklists, 0, 0, 1, 1);

        label_patients = new QLabel(centralWidget);
        label_patients->setObjectName(QStringLiteral("label_patients"));

        gridLayout->addWidget(label_patients, 0, 1, 1, 1);

        WorklistView = new QTreeView(centralWidget);
        WorklistView->setObjectName(QStringLiteral("WorklistView"));
        WorklistView->setMinimumSize(QSize(300, 0));
        WorklistView->setMaximumSize(QSize(300, 16777215));
        WorklistView->setRootIsDecorated(true);
        WorklistView->setHeaderHidden(true);

        gridLayout->addWidget(WorklistView, 1, 0, 5, 1);

        PatientView = new QTableView(centralWidget);
        PatientView->setObjectName(QStringLiteral("PatientView"));
        PatientView->horizontalHeader()->setCascadingSectionResizes(true);
        PatientView->horizontalHeader()->setProperty("showSortIndicator", QVariant(false));
        PatientView->horizontalHeader()->setStretchLastSection(true);
        PatientView->verticalHeader()->setVisible(false);
        PatientView->verticalHeader()->setDefaultSectionSize(33);
        PatientView->verticalHeader()->setMinimumSectionSize(20);

        gridLayout->addWidget(PatientView, 1, 1, 1, 1);

        label_studies = new QLabel(centralWidget);
        label_studies->setObjectName(QStringLiteral("label_studies"));

        gridLayout->addWidget(label_studies, 2, 1, 1, 1);

        StudyView = new QTableView(centralWidget);
        StudyView->setObjectName(QStringLiteral("StudyView"));
        StudyView->verticalHeader()->setVisible(false);

        gridLayout->addWidget(StudyView, 3, 1, 1, 1);

        label_images = new QLabel(centralWidget);
        label_images->setObjectName(QStringLiteral("label_images"));

        gridLayout->addWidget(label_images, 4, 1, 1, 1);

        ImageView = new QListView(centralWidget);
        ImageView->setObjectName(QStringLiteral("ImageView"));
        ImageView->setIconSize(QSize(200, 200));
        ImageView->setResizeMode(QListView::Fixed);
        ImageView->setViewMode(QListView::IconMode);

        gridLayout->addWidget(ImageView, 5, 1, 1, 1);

        WorklistWindowLayout->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(WorklistWindowLayout);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1113, 24));
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
