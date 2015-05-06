#ifndef PATHOLOGYWORKSTATION_H
#define PATHOLOGYWORKSTATION_H

#include <QMainWindow>
#include <QDir>
#include <QStringList>
#include <memory>

class MultiResolutionImage;
class QHBoxLayout;
class QAction;
class QStatusBar;
class QMenuBar;
class PathologyViewer;
class ImageFilterPluginInterface;
class FilterDockWidget;
class WorkstationExtensionPluginInterface;
class QActionGroup;

class PathologyWorkstation : public QMainWindow
{
  Q_OBJECT

public:
  explicit PathologyWorkstation(QWidget* parent = 0);
  ~PathologyWorkstation();

  void setCacheSize(const unsigned long long& cacheMaxByteSize);

  unsigned long long getCacheSize() const;

signals:
  void newImageLoaded(MultiResolutionImage*, std::string);
  void imageClosed();

private slots:
  void on_actionClose_triggered();
  void on_actionOpen_triggered();
  void onFilterResultUpdateRequested();
  void onFilterResultClearRequested();
  void onAutoUpdateStatusChanged(bool autoUpdate);

private:
  MultiResolutionImage *_img;
  unsigned long long _cacheMaxByteSize;
  
  // Plugins and filters
  QDir _pluginsDir;
  std::vector<std::string> _extensionPluginFileNames;
  std::vector<std::string> _filterPluginFileNames;
  std::vector<std::string> _toolPluginFileNames;
  std::shared_ptr<std::vector<std::shared_ptr<ImageFilterPluginInterface> > > _filters;

  // GUI object
  QAction *actionOpen;
  QAction *actionClose;
  QActionGroup *_toolActions;
  QWidget *centralWidget;
  QHBoxLayout *horizontalLayout_2;
  PathologyViewer *pathologyView;
  QMenuBar *menuBar;
  QMenu *menuFile;
  QMenu *menuEdit;
  QMenu *menuView;
  QToolBar *mainToolBar;
  QStatusBar *statusBar;
  FilterDockWidget* _filterDock;

  // Initialize the GUI
  void initializeDocks();
  void setupUi();
  void retranslateUi();
  void loadPlugins();

  std::vector<std::unique_ptr<WorkstationExtensionPluginInterface> > _extensions;
};

#endif // PATHOLOGYWORKSTATION_H
