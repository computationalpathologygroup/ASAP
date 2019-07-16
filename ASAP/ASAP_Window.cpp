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
#include <QTreeWidget>

#include "ASAP_Window.h"
#include "PathologyViewer.h"
#include "interfaces/interfaces.h"
#include "WSITileGraphicsItemCache.h"
#include "config/ASAPMacros.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"
#include "multiresolutionimageinterface/OpenSlideImage.h"

#ifdef WIN32
const char* ASAP_Window::sharedLibraryExtensions = ".dll";
#else
const char* ASAP_Window::sharedLibraryExtensions = ".so";
#endif

using namespace std;

ASAP_Window::ASAP_Window(QWidget *parent) :
    QMainWindow(parent),
    _cacheMaxByteSize(1000*512*512*3),
    _settings(NULL)
{
	setupUi();
	retranslateUi();
	connect(actionOpen, SIGNAL(triggered(bool)), this, SLOT(on_actionOpen_triggered()));
	connect(actionClose, SIGNAL(triggered(bool)), this, SLOT(on_actionClose_triggered()));
	connect(actionAbout, SIGNAL(triggered(bool)), this, SLOT(on_actionAbout_triggered()));

	connect(&m_document_window_controller_,
		&ASAP::DocumentWindowController::viewerFocusChanged,
		this,
		&ASAP_Window::onViewerFocusChanged);

	m_document_window_controller_.SetCacheSize(_cacheMaxByteSize);
	if (m_view_controller_.HasTool("pan"))
	{
		m_view_controller_.SetActiveTool("pan");
		QList<QAction*> toolButtons = mainToolBar->actions();
		for (QList<QAction*>::iterator it = toolButtons.begin(); it != toolButtons.end(); ++it)
		{
			if ((*it)->objectName() == "pan")
			{
				(*it)->setChecked(true);
			}
		}
	}
	
	_settings = new QSettings(QSettings::IniFormat, QSettings::UserScope, "DIAG", "ASAP", this);
	readSettings();
	QStringList args = QApplication::arguments();
	if (args.size() > 1)
	{
		openFile(args[1], "default");
	}
}

ASAP_Window::~ASAP_Window(void)
{
	on_actionClose_triggered();
	writeSettings();
}

void ASAP_Window::writeSettings(void)
{
  _settings->beginGroup("ASAP");
  _settings->setValue("size", size());
  _settings->setValue("maximized", isMaximized());
  _settings->endGroup();
}

void ASAP_Window::readSettings(void)
{
  _settings->beginGroup("ASAP");
  resize(_settings->value("size", QSize(1037, 786)).toSize());
  if (_settings->value("maximized", false).toBool()) {
    this->setWindowState(Qt::WindowMaximized);
  }
  _settings->endGroup();
}

void ASAP_Window::loadPlugins(void) {
	PathologyViewer* viewer(m_view_controller_.GetMasterViewer());
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
				QAction* toolAction = tool->getToolButton();
              connect(toolAction, SIGNAL(triggered(bool)), viewer, SLOT(changeActiveTool()));
              _toolPluginFileNames.push_back(fileName.toStdString());

			  m_view_controller_.AddTool(tool);
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
				  m_view_controller_.AddTool(tools[i]);
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

void ASAP_Window::onViewerFocusChanged(ASAP::DocumentWindow* window)
{
	m_view_controller_.SetMasterViewer(window->m_view_);
}

void ASAP_Window::closeEvent(QCloseEvent *event) {
  event->accept();
}

void ASAP_Window::on_actionAbout_triggered(void) {
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

void ASAP_Window::on_actionClose_triggered(void)
{
    for (std::vector<std::unique_ptr<WorkstationExtensionPluginInterface> >::iterator it = _extensions.begin(); it != _extensions.end(); ++it) {
      if (!(*it)->canClose()) {
        return;
      }
    }
    emit imageClosed();
    _settings->setValue("currentFile", QString());
    this->setWindowTitle("ASAP");

	/*
	if (!m_documents_.empty())
	{
		m_documents_.erase(m_documents_.begin());
		m_document_window_->m_view_->close();
		statusBar->showMessage("Closed file!", 5);
	}*/
}

void ASAP_Window::openFile(const QString& fileName, const QString& factoryName) {
	statusBar->clearMessage();
	try
	{
		size_t document_id = m_documents_.LoadDocument(boost::filesystem::path(fileName.toStdString()), factoryName.toStdString());
		_settings->setValue("lastOpenendPath", QFileInfo(fileName).dir().path());
		_settings->setValue("currentFile", QFileInfo(fileName).fileName());
		this->setWindowTitle(QString("ASAP - ") + QFileInfo(fileName).fileName());


		ASAP::DocumentInstance instance = m_documents_.GetDocument(document_id);
		m_document_window_->AddDocumentInstance(instance);

		ASAP::DocumentInstance instance2 = m_documents_.GetDocument(document_id);
		window2->AddDocumentInstance(instance2);

		emit newImageLoaded(instance.document->GetImage(), fileName.toStdString());
	}
	catch (const std::runtime_error& e)
	{
		// Implies an error with the actual file. We can utilize this for user feedback.
		statusBar->showMessage(e.what());
	}
	catch (const std::invalid_argument& e)
	{
		// Implies an invalid call with no filename or factory name, we can safely ignore this.
		statusBar->showMessage(e.what());
	}
}

void ASAP_Window::on_actionOpen_triggered(void)
{ 
  QString filterList;
  std::set<std::string> allExtensions = MultiResolutionImageFactory::getAllSupportedExtensions();
  QString defaultString = "All supported types (";
  for (auto it = allExtensions.begin(); it != allExtensions.end(); ++it) {
    defaultString += " *." + QString::fromStdString(*it);
  }
  defaultString += ")";
  filterList += defaultString;

  std::vector<std::pair<std::string, std::set<std::string>> > factoriesAndExtensions = MultiResolutionImageFactory::getLoadedFactoriesAndSupportedExtensions();
  for (auto it = factoriesAndExtensions.begin(); it != factoriesAndExtensions.end(); ++it) {
    QString extensionString = "(*." + QString::fromStdString(*(it->second.begin()));
    for (auto extensionIt = std::next(it->second.begin(), 1); extensionIt != it->second.end(); ++extensionIt) {
      extensionString += " *." + QString::fromStdString(*extensionIt);
    }
    extensionString += ")";
    filterList += (";;" + QString::fromStdString(it->first) + " " + extensionString);
  }
  QString selectedFilter;
  QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), _settings->value("lastOpenendPath", QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)).toString(), filterList, &selectedFilter);
  QString selectedFactory = selectedFilter.split("(")[0].trimmed();
  openFile(fileName, selectedFactory == "All supported types" ? "default": selectedFactory);
}

void ASAP_Window::setCacheSize(const unsigned long long& cacheMaxByteSize)
{
	m_document_window_controller_.SetCacheSize(cacheMaxByteSize);
}
    
unsigned long long ASAP_Window::getCacheSize(void) const
{
	return m_document_window_controller_.GetCacheSize();
}

void ASAP_Window::setupUi(void)
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

  m_document_window_ = m_document_window_controller_.SpawnWindow(centralWidget);
  window2 = m_document_window_controller_.SpawnWindow(centralWidget);
  horizontalLayout_2->addWidget(m_document_window_);
  horizontalLayout_2->addWidget(window2);
  this->setCentralWidget(centralWidget);
}

void ASAP_Window::retranslateUi()
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
