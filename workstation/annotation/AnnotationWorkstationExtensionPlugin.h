#ifndef ANNOTATIONWORKSTATIONEXTENSIONPLUGIN_H
#define ANNOTATIONWORKSTATIONEXTENSIONPLUGIN_H

#include "../interfaces/interfaces.h"
#include "config/pathology_config.h"
#include <memory>
#include <QString>
#include <QMap>

class AnnotationService;
class QtAnnotation;
class QtAnnotationGroup;
class QDockWidget;
class QTreeWidget;
class QTreeWidgetItem;
class QObject;
class QEvent;
class QSettings;
class QFrame;
class QLabel;

class EXPORT_PATHOLOGYANNOTATIONPLUGIN AnnotationWorkstationExtensionPlugin : public WorkstationExtensionPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "Diag.PathologyWorkstation.AnnotationWorkstationExtensionPlugin/1.0")
    Q_INTERFACES(WorkstationExtensionPluginInterface)

public :
    bool initialize(PathologyViewer* viewer);
    std::vector<std::shared_ptr<ToolPluginInterface> > getTools();
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
    void deleteAnnotationGroup(QtAnnotationGroup* group);
    void clearSelection();

public slots:
    void onNewImageLoaded(std::weak_ptr<MultiResolutionImage> img, std::string fileName);
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
    void updateAnnotationToolTip(QtAnnotation* annotation);
    void updateGeneratingAnnotationLabel(QtAnnotation* annotation);

private :
    std::vector<std::shared_ptr<ToolPluginInterface> > _annotationTools;
    std::unique_ptr<AnnotationService> _annotationService;
    QtAnnotation* _generatedAnnotation;
    QtAnnotation* _activeAnnotation;
    QSet<QtAnnotation*> _selectedAnnotations;
    QList<QtAnnotation*> _qtAnnotations;
    QMap<QtAnnotation*, QTreeWidgetItem*> _annotToItem;
    QList<QtAnnotationGroup*> _qtAnnotationGroups;
    QDockWidget* _dockWidget;
    QTreeWidget* _treeWidget;
    QLabel* _currentAnnotationHeaderLabel;
    QLabel* _currentAnnotationLabel;
    QFrame* _currentAnnotationLine;
    QEvent* _oldEvent;
    std::weak_ptr<MultiResolutionImage> _img;
    float _currentPixelArea;

    void clearTreeWidget();
    void clearAnnotationList();
    void clearQtAnnotations();

    static unsigned int _annotationIndex;
    static unsigned int _annotationGroupIndex;
};

#endif
