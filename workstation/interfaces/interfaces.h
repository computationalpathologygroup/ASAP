#ifndef INTERFACES_H
#define INTERFACES_H

#include <QtPlugin>
#include <QMouseEvent>
#include <QIcon>
#include <QWidget>
#include <QMutex>

#include <memory>
#include <vector>
#include <string>

#include "core/Patch.h"
#include "imgproc/generic/FilterBase.h"

class QToolBar;
class QDockWidget;
class PathologyViewer;
class QAction;
class QMenu;
class PathologyViewer;
class ProgressMonitor;
class MultiResolutionImage;

class ImageFilterPluginInterface : public QObject
{

public:
    ImageFilterPluginInterface() :
      _settingsPanel(),
      _filter(NULL)
    {}

    virtual ~ImageFilterPluginInterface() {
      _mutex.lock();
      if (_settingsPanel) {
        delete _settingsPanel;
      }
      if (_filter) {
        delete _filter;
      }
      _mutex.unlock();
    }

    virtual QString name() const = 0;
    virtual bool initialize(const ImageSource* image) {
      return true;
    };
    virtual QPointer<QWidget> getSettingsPanel() {
      return _settingsPanel;
    }
    virtual void filter(const Patch<double>& input, QVariant& output) = 0;
    
    void cancel() {
      if (_filter) {
        _filter->cancel();
      }
    };

    void setProgressMonitor(ProgressMonitor* monitor) {
      if (_filter) {
        _filter->setProgressMonitor(monitor);
      }
    }
    virtual ImageFilterPluginInterface* clone() const = 0;
    virtual QIcon icon() const {
      return QIcon(QPixmap(256, 256));
    }

signals:
    virtual void filterParametersChanged() = 0;

protected:
    QPointer<QWidget> _settingsPanel;
    FilterBase* _filter;
    QMutex _mutex;
};

class ToolPluginInterface : public QObject {
public :
  ToolPluginInterface() : _button(NULL), _viewer(NULL)
  {
  }

  virtual ~ToolPluginInterface() { 
    _viewer = NULL;
    _button = NULL;
  }

  virtual std::string name() = 0;
  void setViewer(PathologyViewer* viewer) { _viewer = viewer; }
  virtual void mouseMoveEvent(QMouseEvent *event) { event->ignore(); };
  virtual void mousePressEvent(QMouseEvent *event) { event->ignore(); };
  virtual void mouseReleaseEvent(QMouseEvent *event) { event->ignore(); };
  virtual void mouseDoubleClickEvent(QMouseEvent *event) { event->ignore(); };
  virtual void keyPressEvent(QKeyEvent *event) { event->ignore(); };
  virtual QAction* getToolButton() = 0;

protected :
  PathologyViewer* _viewer;
  QAction* _button;
};

class WorkstationExtensionPluginInterface : public QObject {
public :
  virtual ~WorkstationExtensionPluginInterface() {}
  virtual bool initialize(PathologyViewer* viewer) = 0;
  virtual QToolBar* getToolBar() { return NULL;}
  virtual QMenu* getMenu() { return NULL; }
  virtual QDockWidget* getDockWidget() { return NULL; }
  virtual std::vector<ToolPluginInterface*> getTools() { return std::vector<ToolPluginInterface*>(); }

protected:
  PathologyViewer* _viewer;

public slots:
  virtual void onNewImageLoaded(MultiResolutionImage* img, std::string fileName) {};
  virtual void onImageClosed() {};
};

Q_DECLARE_METATYPE(std::shared_ptr<ImageFilterPluginInterface>)
Q_DECLARE_INTERFACE(ImageFilterPluginInterface, "Diag.PathologyWorkstation.ImageFilterPluginInterface/1.0")
Q_DECLARE_INTERFACE(ToolPluginInterface, "Diag.PathologyWorkstation.ToolPluginInterface/1.0")
Q_DECLARE_INTERFACE(WorkstationExtensionPluginInterface, "Diag.PathologyWorkstation.WorkstationExtensionPluginInterface/1.0")

#endif