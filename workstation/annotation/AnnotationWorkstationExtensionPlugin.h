#ifndef ANNOTATIONWORKSTATIONEXTENSIONPLUGIN_H
#define ANNOTATIONWORKSTATIONEXTENSIONPLUGIN_H

#include "../interfaces/interfaces.h"
#include "config/pathology_config.h"
#include <memory>
#include <QString>
#include <QMap>

class AnnotationService;
class QtAnnotation;
class AnnotationGroup;
class QDockWidget;
class QTreeWidget;
class QTreeWidgetItem;
class QObject;
class QEvent;
class QSettings;

class EXPORT_PATHOLOGYANNOTATIONPLUGIN AnnotationWorkstationExtensionPlugin : public WorkstationExtensionPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "Diag.PathologyWorkstation.AnnotationWorkstationExtensionPlugin/1.0")
    Q_INTERFACES(WorkstationExtensionPluginInterface)

public :
    bool initialize(PathologyViewer* viewer);
    std::vector<ToolPluginInterface*> getTools();
    AnnotationWorkstationExtensionPlugin();
    ~AnnotationWorkstationExtensionPlugin();
    void startAnnotation(float x, float y, const std::string& type);
    void finishAnnotation(bool cancel = false);
    QDockWidget* getDockWidget();
    QtAnnotation* getActiveAnnotation();
    QtAnnotation* getGeneratedAnnotation();
    QSet<QtAnnotation*> getSelectedAnnotations();
    void addAnnotationToSelection(QtAnnotation* annotation);
    void removeAnnotationFromSelection(QtAnnotation* annotation);
    bool eventFilter(QObject* watched, QEvent* event);
    void deleteAnnotation(QtAnnotation* annotation);
    void deleteAnnotationGroup(std::shared_ptr<AnnotationGroup> group);
    void clearSelection();

public slots:
    void onNewImageLoaded(MultiResolutionImage* img, std::string fileName);
    void onImageClosed();
    void addAnnotationGroup();
    void onClearButtonPressed();
    void onLoadButtonPressed(const std::string& filePath = std::string());
    void onSaveButtonPressed();
    void onItemNameChanged(QTreeWidgetItem* item, int column);
    void onTreeWidgetItemDoubleClicked(QTreeWidgetItem * item, int column);
    void onTreeWidgetSelectedItemsChanged();

private slots:
    void resizeOnExpand();

private :
    std::vector<ToolPluginInterface*> _annotationTools;
    AnnotationService* _annotationService;
    QtAnnotation* _generatedAnnotation;
    QtAnnotation* _activeAnnotation;
    QSet<QtAnnotation*> _selectedAnnotations;
    QMap<QString, QtAnnotation*> _qtAnnotations;
    QMap<QString, std::shared_ptr<AnnotationGroup> > _qtAnnotationGroups;
    QDockWidget* _dockWidget;
    QTreeWidget* _treeWidget;
    QEvent* _oldEvent;
    MultiResolutionImage* _img;

    void clearTreeWidget();
    void clearAnnotationList();
    void clearQtAnnotations();

    static unsigned int _annotationIndex;
    static unsigned int _annotationGroupIndex;
    QSettings* _settings;
};

#endif
