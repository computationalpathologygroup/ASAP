#ifndef FILTERWORKSTATIONEXTENSIONPLUGIN_H
#define FILTERWORKSTATIONEXTENSIONPLUGIN_H

#include "../interfaces/interfaces.h"
#include "config/pathology_config.h"

class FilterDockWidget;
class QGraphicsItem;
class FilterThread;
class MultiResolutionImage;

class EXPORT_FILTEREXTENSIONNPLUGIN FilterWorkstationExtensionPlugin : public WorkstationExtensionPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "Diag.PathologyWorkstation.FilterWorkstationExtensionPlugin/1.0")
    Q_INTERFACES(WorkstationExtensionPluginInterface)

private :
  FilterDockWidget* _dockWidget;
  QGraphicsItem* _filterResult;
  FilterThread* _filterThread;
  bool _autoUpdate;
  std::weak_ptr<MultiResolutionImage> _img;

public :
    bool initialize(PathologyViewer* viewer);
    FilterWorkstationExtensionPlugin();
    ~FilterWorkstationExtensionPlugin();
    QDockWidget* getDockWidget();

public slots:
    void onNewImageLoaded(std::weak_ptr<MultiResolutionImage> img, std::string fileName);
    void onImageClosed();
    void updateFilteredImage(QGraphicsItem* result, QRectF size);
    void onChangeCurrentFilter(std::shared_ptr<ImageFilterPluginInterface> filter);
    void onFieldOfViewChanged(const QRectF& FOV, const unsigned int level);

private slots:
    void onFilterResultClearRequested();
    void onFilterResultUpdateRequested();
    void onAutoUpdateStatusChanged(bool autoUpdate);
};

#endif