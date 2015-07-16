#ifndef VISUALIZATIONWORKSTATIONEXTENSIONPLUGIN_H
#define VISUALIZATIONWORKSTATIONEXTENSIONPLUGIN_H

#include "../interfaces/interfaces.h"
#include "config/pathology_config.h"

class QCheckBox;

class EXPORT_VISUALIZATIONPLUGIN VisualizationWorkstationExtensionPlugin : public WorkstationExtensionPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "Diag.PathologyWorkstation.VisualizationWorkstationExtensionPlugin/1.0")
    Q_INTERFACES(WorkstationExtensionPluginInterface)

private :
  MultiResolutionImage* _foreground;
  QDockWidget* _dockWidget;
  QCheckBox* _likelihoodCheckBox;
  float _opacity;

public :
    bool initialize(PathologyViewer* viewer);
    VisualizationWorkstationExtensionPlugin();
    ~VisualizationWorkstationExtensionPlugin();
    QDockWidget* getDockWidget();

public slots:
    void onNewImageLoaded(MultiResolutionImage* img, std::string fileName);
    void onImageClosed();
    void onEnableLikelihoodToggled(bool toggled);
    void onOpacityChanged(double opacity);
    void onEnableSegmentationToggled(bool toggled);

signals: 
    void changeForegroundImage(MultiResolutionImage*);
};

#endif