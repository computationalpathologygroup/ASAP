#ifndef ANNOTATIONWORKSTATIONEXTENSIONPLUGIN_H
#define ANNOTATIONWORKSTATIONEXTENSIONPLUGIN_H

#include "interfaces/interfaces.h"
#include "annotationplugin_export.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <QString>
#include <QMap>

class Annotation;
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

struct QtAnnotationItems
{
	QList<QtAnnotation*>		annotations;
	QList<QtAnnotationGroup*>	groups;
};

class AnnotationPluginInfo : public ASAP::PluginInformation
{
	public:
		AnnotationPluginInfo(void);
		bool								initial_file_loaded;
		unsigned int						annotation_index;
		unsigned int						annotation_group_index;
		std::unique_ptr<AnnotationService>	annotation_service;
};

class ANNOTATIONPLUGIN_EXPORT AnnotationWorkstationExtensionPlugin : public WorkstationExtensionPluginInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ASAP.AnnotationWorkstationExtensionPlugin/1.0")
    Q_INTERFACES(WorkstationExtensionPluginInterface)

public :
	AnnotationWorkstationExtensionPlugin();
	~AnnotationWorkstationExtensionPlugin();

    bool initialize(ASAP::PathologyViewController& controller);
    std::vector<std::shared_ptr<ToolPluginInterface> > getTools();
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
    std::weak_ptr<MultiResolutionImage> getCurrentImage();
    void clearSelection();
    bool canClose(ASAP::DocumentInstance& instance);

public slots:
	void onDocumentChange(ASAP::DocumentInstance* instance);
    void addAnnotationGroup();
    void onClearButtonPressed();
    void onOptionsButtonPressed();
    void onLoadButtonPressed(const std::string& filePath = std::string());
    bool onSaveButtonPressed();
    void onItemNameChanged(QTreeWidgetItem* item, int column);
    void onTreeWidgetItemDoubleClicked(QTreeWidgetItem * item, int column);
    void onTreeWidgetSelectedItemsChanged();

protected:
	virtual void prepareForViewerChange_(void);
	virtual void setupNewViewer_(void);

private slots:
    void resizeOnExpand();
    void updateAnnotationToolTip(QtAnnotation* annotation);
    void updateGeneratingAnnotationLabel(QtAnnotation* annotation);
	void removeViewer(void);

private :
    std::vector<std::shared_ptr<ToolPluginInterface>> _annotationTools;
    QtAnnotation* _generatedAnnotation;
    QtAnnotation* _activeAnnotation;
    QSet<QtAnnotation*> _selectedAnnotations;
    QMap<QtAnnotation*, QTreeWidgetItem*> _annotToItem;
    QDockWidget* _dockWidget;
    QTreeWidget* _treeWidget;
    QLabel* _currentAnnotationHeaderLabel;
    QLabel* _currentAnnotationLabel;
    QFrame* _currentAnnotationLine;
    QEvent* _oldEvent;

	std::mutex m_mutex_;
	std::unordered_map<PathologyViewer*, QtAnnotationItems> m_viewer_ui_items_;

	QtAnnotationItems*		m_annotation_items_;
	PathologyViewer*		m_viewer_;
	ASAP::DocumentInstance*	m_instance_;
	AnnotationPluginInfo*	m_instance_information_;


    float _currentPixelArea;

    bool shouldClear(AnnotationPluginInfo* info);
    void clear(const bool clear_instance_list, const bool delete_from_scene);
    void clearTreeWidget();
    void clearAnnotationList();
    void clearQtAnnotations(const bool delete_from_scene);
	void addAnnotationToGUI_(QtAnnotation* annotation);
	void addAnnotationGroupToGUI_(QtAnnotationGroup* group);

	QtAnnotation* InitializeAnnotation_(std::shared_ptr<Annotation> annotation);
};
#endif