#include "AnnotationWorkstationExtensionPlugin.h"
#include "DotAnnotationTool.h"
#include "PolyAnnotationTool.h"
#include "SplineAnnotationTool.h"
#include "AnnotationService.h"
#include "AnnotationList.h"
#include "AnnotationGroup.h"
#include "QtAnnotation.h"
#include "Annotation.h"
#include "DotQtAnnotation.h"
#include "PolyQtAnnotation.h"
#include "../PathologyViewer.h"
#include <QtUiTools>
#include <QDockWidget>
#include <QTreeWidget>
#include <QPushButton>
#include <QFileDialog>
#include <QApplication>
#include "core/filetools.h"

#include <numeric>
#include <iostream>

unsigned int AnnotationWorkstationExtensionPlugin::_annotationIndex = 0;
unsigned int AnnotationWorkstationExtensionPlugin::_annotationGroupIndex = 0;

AnnotationWorkstationExtensionPlugin::AnnotationWorkstationExtensionPlugin() :
  WorkstationExtensionPluginInterface(),
  _annotationService(NULL),
  _generatedAnnotation(NULL),
  _activeAnnotation(NULL),
  _dockWidget(NULL),
  _treeWidget(NULL),
  _oldEvent(NULL)
{
  QUiLoader loader;
  QFile file(":/AnnotationDockWidget.ui");
  file.open(QFile::ReadOnly);
  _dockWidget = qobject_cast<QDockWidget*>(loader.load(&file));
  if (_dockWidget) {
    _dockWidget->setEnabled(false);
    _treeWidget = _dockWidget->findChild<QTreeWidget*>("AnnotationTreeWidget");
    _treeWidget->viewport()->installEventFilter(this);
    _treeWidget->installEventFilter(this);
    QPushButton* groupButton = _dockWidget->findChild<QPushButton*>("addGroupButton");
    QPushButton* clearButton = _dockWidget->findChild<QPushButton*>("clearButton");
    QPushButton* saveButton = _dockWidget->findChild<QPushButton*>("saveButton");
    QPushButton* loadButton = _dockWidget->findChild<QPushButton*>("loadButton");
    connect(groupButton, SIGNAL(clicked()), this, SLOT(addAnnotationGroup()));
    connect(clearButton, SIGNAL(clicked()), this, SLOT(onClearButtonPressed()));
    connect(saveButton, SIGNAL(clicked()), this, SLOT(onSaveButtonPressed()));
    connect(loadButton, SIGNAL(clicked()), this, SLOT(onLoadButtonPressed()));
    connect(_treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(onItemNameChanged(QTreeWidgetItem*, int)));
    connect(_treeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(onTreeWidgetItemDoubleClicked(QTreeWidgetItem*, int)));
    connect(_treeWidget, SIGNAL(currentItemChanged(QTreeWidgetItem*, QTreeWidgetItem*)), this, SLOT(onTreeWidgetCurrentItemChanged(QTreeWidgetItem*, QTreeWidgetItem*)));
  }
}

AnnotationWorkstationExtensionPlugin::~AnnotationWorkstationExtensionPlugin() {
  onClearButtonPressed();
  if (_annotationService) {
    delete _annotationService;
    _annotationService = NULL;
  }
}

void AnnotationWorkstationExtensionPlugin::onClearButtonPressed() {
  clearSelection();
  clearTreeWidget();
  clearQtAnnotations();
  clearAnnotationList();
  _annotationGroupIndex = 0;
  _annotationIndex = 0;
}

void AnnotationWorkstationExtensionPlugin::clearTreeWidget() {
  if (_treeWidget) {
    _treeWidget->clear();
  }
}

void AnnotationWorkstationExtensionPlugin::clearAnnotationList() {
  if (_annotationService) {
    _annotationService->getList()->removeAllAnnotations();
    _annotationService->getList()->removeAllGroups();
  }
}

void AnnotationWorkstationExtensionPlugin::clearQtAnnotations() {
  for (QMap<QString, QtAnnotation*>::iterator it = _qtAnnotations.begin(); it != _qtAnnotations.end(); ++it) {
    _viewer->scene()->removeItem(it.value());
    (it.value())->deleteLater();
  }
  _qtAnnotations.clear();
  _qtAnnotationGroups.clear();
}

void AnnotationWorkstationExtensionPlugin::onItemNameChanged(QTreeWidgetItem* item, int column) {
  if (item && column == 0) {
    if (item->data(0, Qt::UserRole).value<QString>().contains(QString::fromStdString("_annotation"))) {
      _qtAnnotations[item->data(0, Qt::UserRole).value<QString>()]->getAnnotation()->setName(item->text(0).toStdString());
    }
    else if (item->data(0, Qt::UserRole).value<QString>().contains(QString::fromStdString("_group"))) {
      _qtAnnotationGroups[item->data(0, Qt::UserRole).value<QString>()]->setName(item->text(0).toStdString());
    }
  }
}

void AnnotationWorkstationExtensionPlugin::onTreeWidgetItemDoubleClicked(QTreeWidgetItem * item, int column)
{
  if (_treeWidget && column == 0) {
    _treeWidget->editItem(item, column);
  }
}

void AnnotationWorkstationExtensionPlugin::onTreeWidgetCurrentItemChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous) {
  clearSelection();
  if (current) {
    if (current->data(0, Qt::UserRole).value<QString>().endsWith("_annotation")) {
      addAnnotationToSelection(_qtAnnotations[current->data(0, Qt::UserRole).value<QString>()]);
    }
    else {
      if (current->childCount() > 0) {
        QTreeWidgetItemIterator it(current->child(0));
        while (*it && (*it)->parent() != current->parent()) {
          if ((*it)->data(0, Qt::UserRole).value<QString>().endsWith("_annotation")) {
            addAnnotationToSelection(_qtAnnotations[(*it)->data(0, Qt::UserRole).value<QString>()]);
          }
          ++it;
        }
      }
    }
  }
}

void AnnotationWorkstationExtensionPlugin::onLoadButtonPressed(const std::string& filePath) {
  QString fileName;
  if (filePath.empty()) {
    fileName = QFileDialog::getOpenFileName(NULL, tr("Load annotations"), "D:\\Temp", tr("XML file (*.xml)"));
  }
  else {
    fileName = QString::fromStdString(filePath);
  }
  if (!fileName.isEmpty()) {
    onClearButtonPressed();
    _annotationService->setRepositoryFromSourceFile(fileName.toStdString());
    _annotationService->load();

    // Add loaded groups to treewidget
    std::map<std::string, AnnotationGroup*> childGroups;
    std::map<AnnotationGroup*, QTreeWidgetItem*> annotToWidget;
    std::vector<AnnotationGroup*> grps = _annotationService->getList()->getGroups();
    for (std::vector<AnnotationGroup*>::const_iterator it = grps.begin(); it != grps.end(); ++it) {
      if ((*it)->getGroup() == NULL) {
        std::string key = "Annotation Group " + QString::number(_annotationGroupIndex).toStdString() + "_group";
        _annotationGroupIndex += 1;
        _qtAnnotationGroups[QString::fromStdString(key)] = (*it);
        QTreeWidgetItem* newAnnotationGroup = new QTreeWidgetItem(_treeWidget);
        newAnnotationGroup->setText(0, QString::fromStdString((*it)->getName()));
        newAnnotationGroup->setText(1, "Group");
        newAnnotationGroup->setData(0, Qt::UserRole, QString::fromStdString(key));
        newAnnotationGroup->setFlags(newAnnotationGroup->flags() | Qt::ItemIsEditable);
        annotToWidget[(*it)] = newAnnotationGroup;
      }
      else {
        childGroups[(*it)->getName()] = (*it);
      }
    }
    while (!childGroups.empty()) {
      for (std::map<std::string, AnnotationGroup*>::iterator it = childGroups.begin(); it != childGroups.end();) {
        if (annotToWidget.find((*it).second->getGroup()) != annotToWidget.end()) {
          std::string key = "Annotation Group " + QString::number(_annotationGroupIndex).toStdString() + "_group";
          _annotationGroupIndex += 1;
          _qtAnnotationGroups[QString::fromStdString(key)] = (*it).second;
          QTreeWidgetItem* newAnnotationGroup = new QTreeWidgetItem(annotToWidget[(*it).second->getGroup()]);
          newAnnotationGroup->setText(0, QString::fromStdString((*it).second->getName()));
          newAnnotationGroup->setText(1, "Group");
          newAnnotationGroup->setData(0, Qt::UserRole, QString::fromStdString(key));
          newAnnotationGroup->setFlags(newAnnotationGroup->flags() | Qt::ItemIsEditable);
          annotToWidget[(*it).second] = newAnnotationGroup;
          it = childGroups.erase(it);
        }
        else{
          ++it;
        }
      }
    }
    std::vector<Annotation*> annots = _annotationService->getList()->getAnnotations();
    for (std::vector<Annotation*>::const_iterator it = annots.begin(); it != annots.end(); ++it) {
      QTreeWidgetItem* prnt = _treeWidget->invisibleRootItem();
      if ((*it)->getGroup()) {
        prnt = annotToWidget[(*it)->getGroup()];
      }
      std::string key = "Annotation " + QString::number(_annotationIndex).toStdString() + "_annotation";
      
      // Add QtAnnotation
      QtAnnotation* annot = NULL;
      if ((*it)->getType() == Annotation::Type::DOT) {
        annot = new DotQtAnnotation((*it), _viewer->getSceneScale());
      }
      else if ((*it)->getType() == Annotation::Type::POLYGON) {
        annot = new PolyQtAnnotation((*it), _viewer->getSceneScale());
        dynamic_cast<PolyQtAnnotation*>(annot)->setInterpolationType("linear");
      }
      else if ((*it)->getType() == Annotation::Type::SPLINE) {
        annot = new PolyQtAnnotation((*it), _viewer->getSceneScale());
        dynamic_cast<PolyQtAnnotation*>(annot)->setInterpolationType("spline");
      }
      if (annot) {
        annot->finish();
        _qtAnnotations[QString::fromStdString(key)] = annot;
        annot->setZValue(std::numeric_limits<float>::max());
        _viewer->scene()->addItem(annot);
      }

      _annotationIndex += 1;
      QTreeWidgetItem* newAnnotation = new QTreeWidgetItem(prnt);
      newAnnotation->setText(0, QString::fromStdString((*it)->getName()));
      newAnnotation->setText(1, QString::fromStdString((*it)->getTypeAsString()));
      newAnnotation->setFlags(newAnnotation->flags() & ~Qt::ItemIsDropEnabled);
      newAnnotation->setFlags(newAnnotation->flags() | Qt::ItemIsEditable);
      newAnnotation->setData(0, Qt::UserRole, QString::fromStdString(key));

    }
    _treeWidget->resizeColumnToContents(0);
  }
}

void AnnotationWorkstationExtensionPlugin::onSaveButtonPressed() {
  QString fileName = QFileDialog::getSaveFileName(NULL, tr("Save annotations"), "D:\\Temp", tr("XML file (*.xml)"));
  _annotationService->setRepositoryFromSourceFile(fileName.toStdString());
  _annotationService->save();
}

bool AnnotationWorkstationExtensionPlugin::eventFilter(QObject* watched, QEvent* event) {
  
  if (qobject_cast<QWidget*>(watched) == _treeWidget->viewport()) {
    if (event->type() == QEvent::Drop) {
      if (event == _oldEvent) {
        return false;
      }
      else {
        _oldEvent = event;
        QApplication::sendEvent(_treeWidget->viewport(), event);
      }
      QTreeWidgetItemIterator it(_treeWidget);
      while (*it) {
        QString UID = (*it)->data(0, Qt::UserRole).value<QString>();
        if (!(*it)->parent()) {
          if (UID.endsWith("_group")) {
            _qtAnnotationGroups[UID]->setGroup(NULL);
          }
          else {
            _qtAnnotations[UID]->getAnnotation()->setGroup(NULL);
          }
        }
        else {
          QString parentUID = (*it)->parent()->data(0, Qt::UserRole).value<QString>();
          QString UID = (*it)->data(0, Qt::UserRole).value<QString>();
          if (UID.endsWith("_group")) {
            _qtAnnotationGroups[UID]->setGroup(_qtAnnotationGroups[parentUID]);
          }
          else {
            _qtAnnotations[UID]->getAnnotation()->setGroup(_qtAnnotationGroups[parentUID]);
          }
        }
        ++it;
      }
      _oldEvent = NULL;
    }
  }
  else if (qobject_cast<QWidget*>(watched) == _treeWidget && event->type() == QEvent::KeyPress) {
    QKeyEvent* kpEvent = dynamic_cast<QKeyEvent*>(event);
    if (kpEvent->key() == Qt::Key::Key_Delete) {
      QTreeWidgetItem* itm = _treeWidget->currentItem();
      if (itm->data(0, Qt::UserRole).value<QString>().endsWith("_group")) {
        deleteAnnotationGroup(_qtAnnotationGroups[itm->data(0, Qt::UserRole).value<QString>()]);
      }
      else {
        deleteAnnotation(_qtAnnotations[itm->data(0, Qt::UserRole).value<QString>()]);
      }
    }
  }
  return QObject::eventFilter(watched, event);
}

void AnnotationWorkstationExtensionPlugin::addAnnotationGroup() {
  if (_treeWidget && _annotationService) {
    AnnotationGroup* grp = new AnnotationGroup();
    grp->setName("Annotation Group " + QString::number(_annotationGroupIndex).toStdString());
    _annotationGroupIndex += 1;
    QString grpUID = QString::fromStdString(grp->getName() + "_group");
    _annotationService->getList()->addGroup(grp);
    _qtAnnotationGroups[grpUID] = grp;
    QTreeWidgetItem* newAnnotationGroup = new QTreeWidgetItem(_treeWidget);
    newAnnotationGroup->setText(0, QString::fromStdString(grp->getName()));
    newAnnotationGroup->setText(1, "Group");
    newAnnotationGroup->setData(0, Qt::UserRole, grpUID);
    newAnnotationGroup->setFlags(newAnnotationGroup->flags() | Qt::ItemIsEditable);
    _treeWidget->resizeColumnToContents(0);
  }
}

QDockWidget* AnnotationWorkstationExtensionPlugin::getDockWidget() {
  return _dockWidget;
}

void AnnotationWorkstationExtensionPlugin::onNewImageLoaded(MultiResolutionImage* img, std::string fileName) {
  if (_dockWidget) {
    _dockWidget->setEnabled(true);
  }
  if (!fileName.empty()) {
    std::string annotationPath = fileName;
    core::changeExtension(annotationPath, "xml");
    onLoadButtonPressed(annotationPath);
  }
}

void AnnotationWorkstationExtensionPlugin::onImageClosed() {
  if (_dockWidget) {
    _dockWidget->setEnabled(false);
  }
  onClearButtonPressed();
}

bool AnnotationWorkstationExtensionPlugin::initialize(PathologyViewer* viewer) {
  _viewer = viewer;
  _annotationTools.push_back(new DotAnnotationTool(this, viewer));
  _annotationTools.push_back(new PolyAnnotationTool(this, viewer));
  _annotationTools.push_back(new SplineAnnotationTool(this, viewer));
  _annotationService = new AnnotationService();
  return true;
}

std::vector<ToolPluginInterface*> AnnotationWorkstationExtensionPlugin::getTools() {
  return _annotationTools;
}

void AnnotationWorkstationExtensionPlugin::startAnnotation(float x, float y, const std::string& type) {
  if (_generatedAnnotation) {
    return;
  }
  Annotation* annot = new Annotation();
  annot->addCoordinate(x / _viewer->getSceneScale(), y / _viewer->getSceneScale());
  if (type == "dotannotation") {
    annot->setType(Annotation::Type::DOT);
    _generatedAnnotation = new DotQtAnnotation(annot, _viewer->getSceneScale());
  }
  else if (type == "polyannotation") {
    annot->setType(Annotation::Type::POLYGON);
    PolyQtAnnotation* temp = new PolyQtAnnotation(annot, _viewer->getSceneScale());
    temp->setInterpolationType("linear");
    _generatedAnnotation = temp;
  }
  else if (type == "splineannotation") {
    annot->setType(Annotation::Type::SPLINE);
    PolyQtAnnotation* temp = new PolyQtAnnotation(annot, _viewer->getSceneScale());
    temp->setInterpolationType("spline");
    _generatedAnnotation = temp;    
  }
  else {
    return;
  }
  if (_generatedAnnotation) {
    _generatedAnnotation->setZValue(std::numeric_limits<float>::max());
    _viewer->scene()->addItem(_generatedAnnotation);
    _activeAnnotation = _generatedAnnotation;
    addAnnotationToSelection(_activeAnnotation);
  }
}

void AnnotationWorkstationExtensionPlugin::finishAnnotation(bool cancel) {
  if (_generatedAnnotation) {
    _generatedAnnotation->finish();
    if (!cancel) {
      _generatedAnnotation->getAnnotation()->setName("Annotation " + QString::number(_annotationIndex).toStdString());
      _annotationIndex += 1;
      QString annotUID = QString::fromStdString(_generatedAnnotation->getAnnotation()->getName() + "_annotation");
      _qtAnnotations[annotUID] = _generatedAnnotation;
      _annotationService->getList()->addAnnotation(_generatedAnnotation->getAnnotation());
      QTreeWidgetItem* newAnnotation = new QTreeWidgetItem(_treeWidget);
      newAnnotation->setText(0, QString::fromStdString(_generatedAnnotation->getAnnotation()->getName()));
      newAnnotation->setText(1, QString::fromStdString(_generatedAnnotation->getAnnotation()->getTypeAsString()));
      newAnnotation->setFlags(newAnnotation->flags() & ~Qt::ItemIsDropEnabled);
      newAnnotation->setFlags(newAnnotation->flags() | Qt::ItemIsEditable);
      newAnnotation->setData(0, Qt::UserRole, annotUID);
      _treeWidget->resizeColumnToContents(0);
      _generatedAnnotation = NULL;
    }
    else {
      removeAnnotationFromSelection(_generatedAnnotation);
      _viewer->scene()->removeItem(_generatedAnnotation);
      delete _generatedAnnotation->getAnnotation();
      _generatedAnnotation->deleteLater();
      _generatedAnnotation = NULL;
      _activeAnnotation = NULL;
    }
  }
}

void AnnotationWorkstationExtensionPlugin::deleteAnnotation(QtAnnotation* annotation) {
  if (annotation) {
    removeAnnotationFromSelection(annotation);
    if (_viewer) {
      _viewer->scene()->removeItem(annotation);
    }
    if (_treeWidget) {
      QTreeWidgetItemIterator it(_treeWidget);
      while (*it) {
        if ((*it)->text(0) == QString::fromStdString(annotation->getAnnotation()->getName())) {
          QString annotUID = (*it)->data(0, Qt::UserRole).value<QString>();
          _qtAnnotations.remove(annotUID);
          delete (*it);
          break;
        }
        ++it;
      }
    }
    if (_annotationService) {
      _annotationService->getList()->removeAnnotation(annotation->getAnnotation()->getName());
    }
    annotation->deleteLater();
  }
}

void AnnotationWorkstationExtensionPlugin::deleteAnnotationGroup(AnnotationGroup* group) {
  if (_treeWidget) {
    QTreeWidgetItemIterator it(_treeWidget);
    while (*it) {
      if ((*it)->text(0) == QString::fromStdString(group->getName())) {
        if ((*it)->childCount() > 0) {
          for (int i = (*it)->childCount() - 1; i >= 0; --i) {
            QTreeWidgetItem* itm = (*it)->child(i);
            if (itm->data(0, Qt::UserRole).value<QString>().endsWith("_group")) {
              deleteAnnotationGroup(_qtAnnotationGroups[itm->data(0, Qt::UserRole).value<QString>()]);
            }
            else {
              deleteAnnotation(_qtAnnotations[itm->data(0, Qt::UserRole).value<QString>()]);
            }
          }
        }
        QString groupUID = (*it)->data(0, Qt::UserRole).value<QString>();
        _qtAnnotationGroups.remove(groupUID);
        delete (*it);
        break;
      }
      ++it;
    }
  }
}

QtAnnotation* AnnotationWorkstationExtensionPlugin::getGeneratedAnnotation() {
  return _generatedAnnotation;
}

QtAnnotation* AnnotationWorkstationExtensionPlugin::getActiveAnnotation() {
  return _activeAnnotation;
}

void AnnotationWorkstationExtensionPlugin::addAnnotationToSelection(QtAnnotation* annotation) {
  annotation->setSelected(true);
  _selectedAnnotations.push_back(annotation);
  _activeAnnotation = annotation;
}

void AnnotationWorkstationExtensionPlugin::removeAnnotationFromSelection(QtAnnotation* annotation) {
  annotation->setSelected(false);
  annotation->clearActiveSeedPoint();
  int index = _selectedAnnotations.indexOf(annotation);
  if (index >= 0) {
    _selectedAnnotations.removeAt(index);
  }
  if (annotation == _activeAnnotation) {
    if (_selectedAnnotations.empty()) {
      _activeAnnotation = NULL;
    }
    else {
      _activeAnnotation = _selectedAnnotations.last();
    }
  }
}

void AnnotationWorkstationExtensionPlugin::clearSelection() {
  for (QList<QtAnnotation*>::iterator it = _selectedAnnotations.begin(); it != _selectedAnnotations.end(); ++it) {
    (*it)->setSelected(false);
    (*it)->clearActiveSeedPoint();
  }
  _selectedAnnotations.clear();
  _activeAnnotation = NULL;
}

QList<QtAnnotation*> AnnotationWorkstationExtensionPlugin::getSelectedAnnotations() {
  return _selectedAnnotations;
}