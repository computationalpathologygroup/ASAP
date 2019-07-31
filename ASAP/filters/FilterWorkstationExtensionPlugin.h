#ifndef FILTERWORKSTATIONEXTENSIONPLUGIN_H
#define FILTERWORKSTATIONEXTENSIONPLUGIN_H

#include "interfaces/interfaces.h"

class FilterDockWidget;
class QGraphicsItem;
class FilterThread;
class MultiResolutionImage;

class FilterWorkstationExtensionPlugin : public WorkstationExtensionPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ASAP.FilterWorkstationExtensionPlugin/1.0")
    Q_INTERFACES(WorkstationExtensionPluginInterface)

private :
  FilterDockWidget* _dockWidget;
  QGraphicsItem* _filterResult;
  FilterThread* _filterThread;
  bool _autoUpdate;

public :
    FilterWorkstationExtensionPlugin();
    ~FilterWorkstationExtensionPlugin();
    QDockWidget* getDockWidget();

public slots:
    void updateFilteredImage(QGraphicsItem* result, QRectF size);
    void onChangeCurrentFilter(std::shared_ptr<ImageFilterPluginInterface> filter);
    void onFieldOfViewChanged(const QRectF& FOV, const unsigned int level);
	void onDocumentChange(ASAP::DocumentInstance* document);

protected:
	void prepareForViewerChange_(void);
	void setupNewViewer_(void);

private slots:
    void onFilterResultClearRequested();
    void onFilterResultUpdateRequested();
    void onAutoUpdateStatusChanged(bool autoUpdate);
};

#endif
