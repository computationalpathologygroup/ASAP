#include <string>
#include <vector>
#include <iostream>

#include <QFileDialog>
#include <QToolButton>
#include <QIcon>
#include <QLabel>
#include <QGraphicsEffect>
#include <QDebug>
#include <QPushButton>
#include <QDockWidget>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QMenu>
#include <QMenuBar>
#include <QStatusBar>
#include <QApplication>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPluginLoader>
#include <QComboBox>
#include <QToolBar>
#include <QStyle>
#include <QActionGroup>
#include <QSettings>
#include <QFileInfo>
#include <QStandardPaths>
#include <QtUiTools>

#include "pathologyworkstation.h"
#include "PathologyViewer.h"
#include "interfaces/interfaces.h"
#include "WSITileGraphicsItemCache.h"
#include "io/multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "io/multiresolutionimageinterface/OpenSlideImage.h"

#ifdef WIN32
const char* PathologyWorkstation::sharedLibraryExtensions = ".dll";
#else
const char* PathologyWorkstation::sharedLibraryExtensions = ".so";
#endif

using namespace std;

PathologyWorkstation::PathologyWorkstation(QWidget *parent) :
    QMainWindow(parent),
    _cacheMaxByteSize(1000*512*512*3),
    _settings(NULL)
{
  setupUi();
  retranslateUi();
  connect(actionOpen, SIGNAL(triggered(bool)), this, SLOT(on_actionOpen_triggered()));
  connect(actionClose, SIGNAL(triggered(bool)), this, SLOT(on_actionClose_triggered()));
  connect(actionAbout, SIGNAL(triggered(bool)), this, SLOT(on_actionAbout_triggered()));

  PathologyViewer* view = this->findChild<PathologyViewer*>("pathologyView");
  this->loadPlugins();
  view->setCacheSize(_cacheMaxByteSize);
  if (view->hasTool("pan")) {
    view->setActiveTool("pan");
    QList<QAction*> toolButtons = mainToolBar->actions();
    for (QList<QAction*>::iterator it = toolButtons.begin(); it != toolButtons.end(); ++it) {
      if ((*it)->objectName() == "pan") {
        (*it)->setChecked(true);
      }
    }
  }  
  view->setEnabled(false);
  _settings = new QSettings(QSettings::IniFormat, QSettings::UserScope, "DIAG", "ASAP", this);
  readSettings();
  QStringList args = QApplication::arguments();
  if (args.size() > 1) {
    openFile(args[1]);
  }
}

void PathologyWorkstation::writeSettings()
{
  _settings->beginGroup("ASAP");
  _settings->setValue("size", size());
  _settings->setValue("maximized", isMaximized());
  _settings->endGroup();
}

void PathologyWorkstation::readSettings()
{
  _settings->beginGroup("ASAP");
  resize(_settings->value("size", QSize(1037, 786)).toSize());
  if (_settings->value("maximized", false).toBool()) {
    this->setWindowState(Qt::WindowMaximized);
  }
  _settings->endGroup();
}

void PathologyWorkstation::loadPlugins() {
  PathologyViewer* viewer = this->findChild<PathologyViewer*>("pathologyView");
  _pluginsDir = QDir(qApp->applicationDirPath());
  if (_pluginsDir.cd("plugins")) {
    if (_pluginsDir.cd("tools")) {
      foreach(QString fileName, _pluginsDir.entryList(QDir::Files)) {
        if (fileName.toLower().endsWith(sharedLibraryExtensions)) {
          QPluginLoader loader(_pluginsDir.absoluteFilePath(fileName));
          QObject *plugin = loader.instance();
          if (plugin) {
            std::shared_ptr<ToolPluginInterface> tool(qobject_cast<ToolPluginInterface*>(plugin));
            if (tool) {
              tool->setViewer(viewer);
              QAction* toolAction = tool->getToolButton();
              connect(toolAction, SIGNAL(triggered(bool)), viewer, SLOT(changeActiveTool()));
              _toolPluginFileNames.push_back(fileName.toStdString());
              viewer->addTool(tool);
              QToolBar* mainToolBar = this->findChild<QToolBar *>("mainToolBar");
              toolAction->setCheckable(true);
              _toolActions->addAction(toolAction);
              mainToolBar->addAction(toolAction);
            }
          }
        }
      }
      _pluginsDir.cdUp();
    }
    if (_pluginsDir.cd("workstationextension")) {
      QDockWidget* lastDockWidget = NULL;
      QDockWidget* firstDockWidget = NULL;
      foreach(QString fileName, _pluginsDir.entryList(QDir::Files)) {
        if (fileName.toLower().endsWith(sharedLibraryExtensions)) {
          QPluginLoader loader(_pluginsDir.absoluteFilePath(fileName));
          QObject *plugin = loader.instance();
          if (plugin) {
            std::unique_ptr<WorkstationExtensionPluginInterface> extension(qobject_cast<WorkstationExtensionPluginInterface*>(plugin));
            if (extension) {
              _extensionPluginFileNames.push_back(fileName.toStdString());
              connect(this, SIGNAL(newImageLoaded(std::weak_ptr<MultiResolutionImage>, std::string)), &*extension, SLOT(onNewImageLoaded(std::weak_ptr<MultiResolutionImage>, std::string)));
              connect(this, SIGNAL(imageClosed()), &*extension, SLOT(onImageClosed()));
              extension->initialize(viewer);
              if (extension->getToolBar()) {
                this->addToolBar(extension->getToolBar());
              }
              if (extension->getDockWidget()) {
                QDockWidget* extensionDW = extension->getDockWidget();
                extensionDW->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
                if (lastDockWidget) {
                  this->tabifyDockWidget(lastDockWidget, extensionDW);
                }
                else {
                  this->addDockWidget(Qt::LeftDockWidgetArea, extensionDW);
                  firstDockWidget = extensionDW;
                }
                extensionDW->setTitleBarWidget(new QWidget());
                lastDockWidget = extensionDW;
                QMenu* viewMenu = this->findChild<QMenu*>("menuView");
                QMenu* viewDocksMenu = viewMenu->findChild<QMenu*>("menuViewDocks");
                if (!viewDocksMenu) {
                  viewDocksMenu = viewMenu->addMenu("Docks");
                  viewDocksMenu->setObjectName("menuViewDocks");
                }
                viewDocksMenu->addAction(extensionDW->toggleViewAction());
              }
              if (extension->getMenu()) {
                this->menuBar->addMenu(extension->getMenu());
              }
              std::vector<std::shared_ptr<ToolPluginInterface> > tools = extension->getTools();
              if (!tools.empty()) {
                mainToolBar->addSeparator();
                for (unsigned int i = 0; i < tools.size(); ++i) {
                  QAction* toolAction = tools[i]->getToolButton();
                  connect(toolAction, SIGNAL(triggered(bool)), viewer, SLOT(changeActiveTool()));
                  viewer->addTool(tools[i]);
                  mainToolBar->addAction(toolAction);
                  toolAction->setCheckable(true);
                  _toolActions->addAction(toolAction);
                }
              }
              _extensions.push_back(std::move(extension));
            }
          }
        }
      }
      _pluginsDir.cdUp();
      if (firstDockWidget) {
        firstDockWidget->raise();
      }
    }
  }
}

PathologyWorkstation::~PathologyWorkstation()
{
  on_actionClose_triggered();
  writeSettings();
}

void PathologyWorkstation::on_actionAbout_triggered() {
  QUiLoader loader;
  QFile file(":/ASAP_ui/aboutdialog.ui");
  file.open(QFile::ReadOnly);
  QDialog* content = qobject_cast<QDialog*>(loader.load(&file, this));
  if (content) {
    QLabel* generalInfoLabel = content->findChild<QLabel*>("generalInfoLabel");
    QString generalInfoText = generalInfoLabel->text();
    generalInfoText.replace("@VERSION_STRING@", ASAP_VERSION_STRING);
    generalInfoLabel->setText(generalInfoText);
    QTreeWidget* pluginList = content->findChild<QTreeWidget*>("loadedPluginsOverviewTreeWidget");
    QList<QTreeWidgetItem*> root_items = pluginList->findItems("Tool", Qt::MatchExactly);
    if (!root_items.empty()) {
      QTreeWidgetItem* root_item = root_items[0];
      for (std::vector<std::string>::const_iterator it = _toolPluginFileNames.begin(); it != _toolPluginFileNames.end(); ++it) {
        root_item->addChild(new QTreeWidgetItem(QStringList(QString::fromStdString(*it))));
      }
    }
    root_items = pluginList->findItems("Workstation Extension", Qt::MatchExactly);
    if (!root_items.empty()) {
      QTreeWidgetItem* root_item = root_items[0];
      for (std::vector<std::string>::const_iterator it = _extensionPluginFileNames.begin(); it != _extensionPluginFileNames.end(); ++it) {
        root_item->addChild(new QTreeWidgetItem(QStringList(QString::fromStdString(*it))));
      }
    }    
    content->exec();
  }
  file.close();
}

void PathologyWorkstation::on_actionClose_triggered()
{
    emit imageClosed();
    _settings->setValue("currentFile", QString());
    this->setWindowTitle("ASAP");
    if (_img) {
		  PathologyViewer* view = this->findChild<PathologyViewer*>("pathologyView");
		  view->close();
		  _img.reset();
		  statusBar->showMessage("Closed file!", 5);
    }
}

void PathologyWorkstation::openFile(const QString& fileName) {
  statusBar->clearMessage();
  if (!fileName.isEmpty()) {
    if (_img) {
      on_actionClose_triggered();
    }
    std::string fn = fileName.toStdString();
    _settings->setValue("lastOpenendPath", QFileInfo(fileName).dir().path());
    _settings->setValue("currentFile", QFileInfo(fileName).fileName());
    this->setWindowTitle(QString("ASAP - ") + QFileInfo(fileName).fileName());
    MultiResolutionImageReader imgReader;
    _img.reset(imgReader.open(fn));
    if (_img) {
      if (_img->valid()) {
        if (std::shared_ptr<OpenSlideImage> openslide_img = dynamic_pointer_cast<OpenSlideImage>(_img)) {
          openslide_img->setIgnoreAlpha(false);
        }
        vector<unsigned long long> dimensions = _img->getLevelDimensions(_img->getNumberOfLevels() - 1);
        PathologyViewer* view = this->findChild<PathologyViewer*>("pathologyView");
        view->initialize(_img);
        emit newImageLoaded(_img, fn);
      }
      else {
        statusBar->showMessage("Unsupported file type version");
      }
    }
    else {
      statusBar->showMessage("Invalid file type");
    }
  }
}

void PathologyWorkstation::on_actionOpen_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), _settings->value("lastOpenendPath", QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)).toString(), tr("Slide files (*.lif *.svs *.mrxs *.ndpi *.tif *.tiff);;All files (*.*)"));
    openFile(fileName);
}

void PathologyWorkstation::setCacheSize(const unsigned long long& cacheMaxByteSize) {
  PathologyViewer* view = this->findChild<PathologyViewer*>("pathologyView");
  if (view) {
    view->setCacheSize(_cacheMaxByteSize);
  }
}
    
unsigned long long PathologyWorkstation::getCacheSize() const {
  PathologyViewer* view = this->findChild<PathologyViewer*>("pathologyView");
  if (view) {
    return view->getCacheSize();
  }
}

void PathologyWorkstation::setupUi()
{
  if (this->objectName().isEmpty()) {
      this->setObjectName(QStringLiteral("ASAP"));
  }
  this->resize(1037, 786);
  this->setTabPosition(Qt::DockWidgetArea::LeftDockWidgetArea, QTabWidget::East);
  this->setTabPosition(Qt::DockWidgetArea::RightDockWidgetArea, QTabWidget::West);
  QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  sizePolicy.setHorizontalStretch(0);
  sizePolicy.setVerticalStretch(0);
  sizePolicy.setHeightForWidth(this->sizePolicy().hasHeightForWidth());
  this->setSizePolicy(sizePolicy);
  actionOpen = new QAction(this);
  actionOpen->setObjectName(QStringLiteral("actionOpen"));
  actionClose = new QAction(this);
  actionClose->setObjectName(QStringLiteral("actionClose"));
  actionOpen->setIcon(QIcon(QPixmap(":/ASAP_icons/open.png")));
  actionClose->setIcon(QIcon(QPixmap(":/ASAP_icons/close.png")));
  actionAbout = new QAction(this);
  actionAbout->setObjectName(QStringLiteral("actionAbout"));
  menuBar = new QMenuBar(this);
  menuBar->setObjectName(QStringLiteral("menuBar"));
  menuBar->setGeometry(QRect(0, 0, 1037, 21));
  menuFile = new QMenu(menuBar);
  menuFile->setObjectName(QStringLiteral("menuFile"));
  menuView = new QMenu(menuBar);
  menuView->setObjectName(QStringLiteral("menuView"));
  menuHelp = new QMenu(menuBar);
  menuHelp->setObjectName(QStringLiteral("menuHelp"));
  this->setMenuBar(menuBar);
  mainToolBar = new QToolBar(this);
  mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
  mainToolBar->addAction(actionOpen);
  mainToolBar->addAction(actionClose);
  mainToolBar->addSeparator();
  this->addToolBar(Qt::TopToolBarArea, mainToolBar);
  _toolActions = new QActionGroup(this);
  statusBar = new QStatusBar(this);
  statusBar->setObjectName(QStringLiteral("statusBar"));
  this->setStatusBar(statusBar);

  menuBar->addAction(menuFile->menuAction());
  menuBar->addAction(menuView->menuAction());
  menuBar->addAction(menuHelp->menuAction());
  menuFile->addAction(actionOpen);
  menuFile->addAction(actionClose);
  menuHelp->addAction(actionAbout);
  centralWidget = new QWidget(this);
  centralWidget->setObjectName(QStringLiteral("centralWidget"));
  sizePolicy.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
  centralWidget->setSizePolicy(sizePolicy);
  centralWidget->setLocale(QLocale(QLocale::English, QLocale::UnitedStates));
  horizontalLayout_2 = new QHBoxLayout(centralWidget);
  horizontalLayout_2->setSpacing(6);
  horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
  horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
  pathologyView = new PathologyViewer(centralWidget);
  pathologyView->setObjectName(QStringLiteral("pathologyView"));

  horizontalLayout_2->addWidget(pathologyView);

  this->setCentralWidget(centralWidget);
}

void PathologyWorkstation::retranslateUi()
{
  this->setWindowTitle(QApplication::translate("PathologyWorkstation", "ASAP", 0));
  actionOpen->setText(QApplication::translate("PathologyWorkstation", "Open", 0));
  actionOpen->setIconText(QApplication::translate("PathologyWorkstation", "Open", 0));
  actionAbout->setText(QApplication::translate("PathologyWorkstation", "About...", 0));
  actionOpen->setShortcut(QApplication::translate("PathologyWorkstation", "Ctrl+O", 0));
  actionClose->setText(QApplication::translate("PathologyWorkstation", "Close", 0));
  actionClose->setShortcut(QApplication::translate("PathologyWorkstation", "Ctrl+C", 0));
  actionClose->setIconText(QApplication::translate("PathologyWorkstation", "Close", 0));
  menuFile->setTitle(QApplication::translate("PathologyWorkstation", "File", 0));
  menuView->setTitle(QApplication::translate("PathologyWorkstation", "View", 0));
  menuHelp->setTitle(QApplication::translate("PathologyWorkstation", "Help", 0));
} 
