#ifndef _FilterContainer
#define _FilterContainer

#include <qdockwidget.h>
#include <vector>
#include <memory>

class QListWidget;
class QListWidgetItem;
class ImageFilterPluginInterface;
class MultiResolutionImage;
class QLayout;
class QProgressBar;
class QLabel;
class QtProgressMonitor;
class QPushButton;
class QCheckBox;

class FilterDockWidget : public QDockWidget {
  Q_OBJECT

private :
  QListWidget* _availableFilters;
  QListWidgetItem* _currentSelection;
  QWidget* _settingsPanel;
  QLayout* _layout;
  QPushButton *_applyFilter;
  QPushButton *_clearFilter;
  QCheckBox *_autoUpdateCheckBox;
  QProgressBar* _progressBar;
  std::shared_ptr<QtProgressMonitor> _monitor;

  bool _autoUpdate;

  void stopProgressTracking();

private slots :
  void onItemClicked(QListWidgetItem* item);
  void onProcessing();
  void onApplyFilterClicked();
  void onClearFilterClicked();
  void onAutoUpdateToggled(bool autoUpdate);
  void onFilterParametersChanged();

public slots :
  void onNewImageLoaded(std::weak_ptr<MultiResolutionImage> img);
  void onImageClosed();

signals:
  void changeCurrentFilter(std::shared_ptr<ImageFilterPluginInterface> filter);
  void requestFilterResultUpdate();
  void requestFilterResultClear();
  void changeAutoUpdateStatus(bool autoUpdate);

public :
  FilterDockWidget(QWidget *parent = 0, Qt::WindowFlags flags = 0);
  ~FilterDockWidget();

};

#endif