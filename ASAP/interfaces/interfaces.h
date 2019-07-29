#ifndef INTERFACES_H
#define INTERFACES_H

#include <QtPlugin>
#include <QMouseEvent>
#include <QIcon>
#include <QWidget>
#include <QMutex>
#include <QAction>

#include <memory>
#include <vector>
#include <string>

#include "core/Patch.h"
#include "imgproc/basicfilters/FilterBase.h"
#include "ASAP/PathologyViewer.h"
#include "ASAP/PathologyViewController.h"

class QSettings;
class QToolBar;
class QDockWidget;
class PathologyViewer;
class QMenu;
class ProgressMonitor;
class MultiResolutionImage;

namespace ASAP
{
	class PathologyViewController;
}

class ASAPLIB_EXPORT ImageFilterPluginInterface : public QObject
{
	Q_OBJECT
	public:
		ImageFilterPluginInterface() : QObject(), _settingsPanel()
		{ }

		virtual ~ImageFilterPluginInterface() {
		  _mutex.lock();
		  if (_settingsPanel) {
			delete _settingsPanel;
		  }
		  _mutex.unlock();
		}

		virtual bool initialize(const ImageSource* image) {
		  return true;
		};

		virtual QPointer<QWidget> getSettingsPanel() {
		  return _settingsPanel;
		}

		void cancel() {
		  if (_filter) {
			_filter->cancel();
		  }
		};

		void setProgressMonitor(std::shared_ptr<ProgressMonitor> monitor) {
		  if (_filter) {
			_filter->setProgressMonitor(monitor);
		  }
		}
		
		virtual QIcon icon() const {
		  return QIcon(QPixmap(256, 256));
		}

		virtual QString name() const = 0;
		virtual void filter(const Patch<double>& input, QVariant& output) = 0;
		virtual ImageFilterPluginInterface* clone() const = 0;

	signals:
		virtual void filterParametersChanged() = 0;

	protected:
		QPointer<QWidget> _settingsPanel;
		std::unique_ptr<FilterBase> _filter;
		QMutex _mutex;
};

class ASAPLIB_EXPORT ToolPluginInterface : public QObject {
	Q_OBJECT
	public:
	  ToolPluginInterface() : QObject(), _button(NULL), _controller(NULL), _active(false)
	  { }

	  virtual ~ToolPluginInterface() {
		_controller = NULL;
		_button = NULL;
	  }
    
	  void setController(ASAP::PathologyViewController& controller) { _controller = &controller; }
	  bool active() { return _active; }
	  virtual void setActive(bool active) { _active = active; }

	  virtual void mouseMoveEvent(QMouseEvent *event) { event->ignore(); };
	  virtual void mousePressEvent(QMouseEvent *event) { event->ignore(); };
	  virtual void mouseReleaseEvent(QMouseEvent *event) { event->ignore(); };
	  virtual void mouseDoubleClickEvent(QMouseEvent *event) { event->ignore(); };
	  virtual void keyPressEvent(QKeyEvent *event) { event->ignore(); };

	  virtual std::string name() = 0;
	  virtual QAction* getToolButton() = 0;

	protected:
	  QPointer<ASAP::PathologyViewController> _controller;
	  QPointer<QAction> _button;
	  bool _active;
};

class ASAPLIB_EXPORT WorkstationExtensionPluginInterface : public QObject {
	Q_OBJECT
	public :
	  WorkstationExtensionPluginInterface() : QObject(), _settings(NULL) {}
	  virtual ~WorkstationExtensionPluginInterface() {}
	  virtual QToolBar* getToolBar() { return NULL;}
	  virtual QMenu* getMenu() { return NULL; }
	  virtual QDockWidget* getDockWidget() { return NULL; }
	  virtual bool canClose(ASAP::DocumentInstance& instance) { return true; }
	  virtual std::vector<std::shared_ptr<ToolPluginInterface> > getTools() { return std::vector<std::shared_ptr<ToolPluginInterface> >(); }

	  virtual bool initialize(ASAP::PathologyViewController& controller);

	public slots:
	  void onViewerChangeStart(void);
	  void onViewerChangeFinished(void);

	  // Explicitly request new implementations to take these into account
	  virtual void onDocumentChange(ASAP::DocumentInstance* document) = 0;

	protected:
	  ASAP::PathologyViewController* _controller;
	  QSettings* _settings;

	  virtual void prepareForViewerChange_(void) = 0;
	  virtual void setupNewViewer_(void) = 0;
};

Q_DECLARE_METATYPE(std::shared_ptr<ImageFilterPluginInterface>)
Q_DECLARE_INTERFACE(ImageFilterPluginInterface, "ASAP.ImageFilterPluginInterface/1.0")
Q_DECLARE_INTERFACE(ToolPluginInterface, "ASAP.ToolPluginInterface/1.0")
Q_DECLARE_INTERFACE(WorkstationExtensionPluginInterface, "ASAP.WorkstationExtensionPluginInterface/1.0")
#endif