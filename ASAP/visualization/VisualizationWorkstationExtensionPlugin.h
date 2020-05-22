#ifndef VISUALIZATIONWORKSTATIONEXTENSIONPLUGIN_H
#define VISUALIZATIONWORKSTATIONEXTENSIONPLUGIN_H

#include <memory>
#include "interfaces/interfaces.h"
#include "core/PathologyEnums.h"

class QCheckBox;
class QDialog;
class XmlRepository;
class QGraphicsPolygonItem;
class AnnotationList;
class QHBoxLayout;

class VisualizationWorkstationExtensionPlugin : public WorkstationExtensionPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ASAP.VisualizationWorkstationExtensionPlugin/1.0")
    Q_INTERFACES(WorkstationExtensionPluginInterface)

private :
  // Images
  std::shared_ptr<MultiResolutionImage> _foreground;
  std::vector<unsigned long long> _backgroundDimensions;

  // GUI Components
  QDockWidget* _dockWidget;
  QCheckBox* _likelihoodCheckBox;
  QCheckBox* _segmentationCheckBox;
  QDialog*  _LUTEditor;
  QWidget* _LUTEditingArea;

  // Visualization parameters
  std::map<std::string, pathology::LUT> _colorLookupTables;
  std::map<std::string, pathology::LUT> _LUTsBeforeEdit;
  QString _currentLUT;
  QString _currentLUTBeforeEdit;
  float _opacity;
  float _foregroundChannel;

  // State parameters
  bool  _renderingEnabled;
  float _foregroundScale;
  bool  _editingLUT;
  bool  _previewingLUT;

  void loadNewForegroundImage(const std::string& resultImagePth);
  void setDefaultVisualizationParameters(std::shared_ptr<MultiResolutionImage> img);
  void updateObjectNames();
  void clearForegroundImage();
  QHBoxLayout* createLUTEntry(const pathology::LUT& currentLUT, int index);

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
    void onOpenResultImageClicked();
    void onLUTChanged(const QString& LUTname);
    void onLUTEntryChanged();
    void onChannelChanged(int channel);

signals: 
    void changeForegroundImage(std::weak_ptr<MultiResolutionImage>, float scale);

private slots:
  void generateLUTEditingWidgets(const QString& currentLUTName);
  void pickLUTColor();
  void onLUTIndexChanged(double newIndex);
  void removeLUTEntry();
  void addLUTEntry();
  void handleEditLUTRequest();
  void updateLUTPreviewStatus(int newCheckedState);
  void resetAllLUTs();
  void addLUT();
  void removeLUT();
  void duplicateLUT();
};

#endif
