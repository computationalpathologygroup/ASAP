#ifndef VISUALIZATIONWORKSTATIONEXTENSIONPLUGIN_H
#define VISUALIZATIONWORKSTATIONEXTENSIONPLUGIN_H

#include <memory>
#include "../interfaces/interfaces.h"
#include "config/pathology_config.h"

class QCheckBox;
class XmlRepository;
class QGraphicsPolygonItem;
class AnnotationList;

class EXPORT_VISUALIZATIONPLUGIN VisualizationWorkstationExtensionPlugin : public WorkstationExtensionPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "Diag.PathologyWorkstation.VisualizationWorkstationExtensionPlugin/1.0")
    Q_INTERFACES(WorkstationExtensionPluginInterface)

private :

  void addSegmentationsToViewer();
  void removeSegmentationsFromViewer();

  std::shared_ptr<MultiResolutionImage> _foreground;
  QDockWidget* _dockWidget;
  QCheckBox* _likelihoodCheckBox;
  QCheckBox* _segmentationCheckBox;
  float _opacity;
  float _foregroundScale;
  std::shared_ptr<XmlRepository> _xmlRepo;
  std::shared_ptr<AnnotationList> _lst;
  QList<QGraphicsPolygonItem*> _polygons;

public :
    bool initialize(PathologyViewer* viewer);
    VisualizationWorkstationExtensionPlugin();
    ~VisualizationWorkstationExtensionPlugin();
    QDockWidget* getDockWidget();    

public slots:
    void onNewImageLoaded(std::weak_ptr<MultiResolutionImage> img, std::string fileName);
    void onImageClosed();
    void onEnableLikelihoodToggled(bool toggled);
    void onOpacityChanged(double opacity);
    void onEnableSegmentationToggled(bool toggled);

signals: 
    void changeForegroundImage(std::weak_ptr<MultiResolutionImage>, float scale);
};

#endif