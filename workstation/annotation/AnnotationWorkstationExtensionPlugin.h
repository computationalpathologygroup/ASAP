#ifndef ANNOTATIONWORKSTATIONEXTENSIONPLUGIN_H
#define ANNOTATIONWORKSTATIONEXTENSIONPLUGIN_H

#include "../interfaces/interfaces.h"
#include "config/pathology_config.h"
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

class EXPORT_PATHOLOGYANNOTATION AnnotationWorkstationExtensionPlugin : public WorkstationExtensionPluginInterface
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
    QList<QtAnnotation*> getSelectedAnnotations();
    void addAnnotationToSelection(QtAnnotation* annotation);
    void removeAnnotationFromSelection(QtAnnotation* annotation);
    void clearSelection();
    QList<QtAnnotation*> getSelectedAnnotation();
    bool eventFilter(QObject* watched, QEvent* event);
    void deleteAnnotation(QtAnnotation* annotation);
    void deleteAnnotationGroup(AnnotationGroup* group);

public slots:
    void onNewImageLoaded(MultiResolutionImage* img, std::string fileName);
    void onImageClosed();
    void addAnnotationGroup();
    void onClearButtonPressed();
    void onLoadButtonPressed();
    void onSaveButtonPressed();
    void onItemNameChanged(QTreeWidgetItem* item, int column);
    void onTreeWidgetItemDoubleClicked(QTreeWidgetItem * item, int column);
    void onTreeWidgetCurrentItemChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous);

private :
    std::vector<ToolPluginInterface*> _annotationTools;
    AnnotationService* _annotationService;
    QtAnnotation* _generatedAnnotation;
    QtAnnotation* _activeAnnotation;
    QList<QtAnnotation*> _selectedAnnotations;
    QMap<QString, QtAnnotation*> _qtAnnotations;
    QMap<QString, AnnotationGroup*> _qtAnnotationGroups;
    QDockWidget* _dockWidget;
    QTreeWidget* _treeWidget;
    QEvent* _oldEvent;

    void clearTreeWidget();
    void clearAnnotationList();
    void clearQtAnnotations();

    static unsigned int _annotationIndex;
    static unsigned int _annotationGroupIndex;
};

#endif
