#include "AnnotationWorkstationExtensionPlugin.h"
#include "DotAnnotationTool.h"
#include "PolyAnnotationTool.h"
#include "SplineAnnotationTool.h"
#include "AnnotationService.h"
#include "AnnotationList.h"
#include "AnnotationGroup.h"
#include "QtAnnotation.h"
#include "Annotation.h"
#include "ImageScopeRepository.h"
#include "AnnotationToMask.h"
#include "DotQtAnnotation.h"
#include "PolyQtAnnotation.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "../PathologyViewer.h"
#include <QtUiTools>
#include <QDockWidget>
#include <QTreeWidget>
#include <QPushButton>
#include <QFileDialog>
#include <QLabel>
#include <QApplication>
#include <QColorDialog>
#include <QSettings>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QSpinBox>
#include "core/filetools.h"
#include "../QtProgressMonitor.h"
#include <QProgressDialog>
#include <QMessageBox>
#include <QInputDialog>

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
  QFile file(":/AnnotationWorkstationExtensionPlugin_ui/AnnotationDockWidget.ui");
  bool openend = file.open(QFile::ReadOnly);
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
    connect(_treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(onTreeWidgetSelectedItemsChanged()));
    connect(_treeWidget, SIGNAL(itemExpanded(QTreeWidgetItem*)), this, SLOT(resizeOnExpand()));
  }
  _settings = new QSettings(QSettings::IniFormat, QSettings::UserScope, "DIAG", "ASAP", this);
}

AnnotationWorkstationExtensionPlugin::~AnnotationWorkstationExtensionPlugin() {
  onClearButtonPressed();
  if (_annotationService) {
    delete _annotationService;
    _annotationService = NULL;
  }
}

void AnnotationWorkstationExtensionPlugin::onClearButtonPressed() {
  if (_generatedAnnotation) {
    PolyQtAnnotation* tmp = dynamic_cast<PolyQtAnnotation*>(_generatedAnnotation);
    if (tmp) {
      if (tmp->getInterpolationType() == "spline") {
        dynamic_cast<SplineAnnotationTool*>(_annotationTools[2])->cancelAnnotation();
      }
      else {
        dynamic_cast<PolyAnnotationTool*>(_annotationTools[1])->cancelAnnotation();
      }
    }
  }
  _treeWidget->clearSelection();
  clearTreeWidget();
  clearQtAnnotations();
  clearAnnotationList();
  _annotationGroupIndex = 0;
  _annotationIndex = 0;
}

void AnnotationWorkstationExtensionPlugin::resizeOnExpand() {
  if (_treeWidget) {
    _treeWidget->resizeColumnToContents(0);
    _treeWidget->resizeColumnToContents(1);
  }
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
  _activeAnnotation = NULL;
  _generatedAnnotation = NULL;
}

void AnnotationWorkstationExtensionPlugin::onItemNameChanged(QTreeWidgetItem* item, int column) {
  if (item && column == 1) {
    if (item->data(1, Qt::UserRole).value<QString>().contains(QString::fromStdString("_annotation"))) {
      _qtAnnotations[item->data(1, Qt::UserRole).value<QString>()]->getAnnotation()->setName(item->text(1).toStdString());
    }
    else if (item->data(1, Qt::UserRole).value<QString>().contains(QString::fromStdString("_group"))) {
      _qtAnnotationGroups[item->data(1, Qt::UserRole).value<QString>()]->setName(item->text(1).toStdString());
    }
  }
}

void AnnotationWorkstationExtensionPlugin::onTreeWidgetItemDoubleClicked(QTreeWidgetItem * item, int column)
{
  if (_treeWidget && column == 1) {
    _treeWidget->editItem(item, column);
  }
  else if (_treeWidget && column == 0) {
    QColor newColor = QColorDialog::getColor(item->data(0, Qt::UserRole).value<QColor>(), NULL, QString("Select a color"));
    if (newColor.isValid()) {
      int cHeight = _treeWidget->visualItemRect(item).height();
      QPixmap iconPM(cHeight, cHeight);
      iconPM.fill(newColor);
      QIcon color(iconPM);
      item->setIcon(0, color);
      item->setData(0, Qt::UserRole, newColor);
      if (item->data(1, Qt::UserRole).value<QString>().contains(QString::fromStdString("_annotation"))) {
        _qtAnnotations[item->data(1, Qt::UserRole).value<QString>()]->getAnnotation()->setColor(newColor.name().toStdString());
      }
      else if (item->data(1, Qt::UserRole).value<QString>().contains(QString::fromStdString("_group"))) {
        _qtAnnotationGroups[item->data(1, Qt::UserRole).value<QString>()]->setColor(newColor.name().toStdString());
      }
    }
  }
}

void AnnotationWorkstationExtensionPlugin::onTreeWidgetSelectedItemsChanged() {
  // First clear all the selected annotations
  for (QSet<QtAnnotation*>::iterator it = _selectedAnnotations.begin(); it != _selectedAnnotations.end(); ++it) {
    (*it)->setSelected(false);
    (*it)->clearActiveSeedPoint();
  }
  _selectedAnnotations.clear();
  _activeAnnotation = NULL;

  // Then update from list view
  QList<QTreeWidgetItem*> selItems = _treeWidget->selectedItems();
  for (QList<QTreeWidgetItem*>::iterator itm = selItems.begin(); itm != selItems.end(); ++itm) {
    if ((*itm)->data(1, Qt::UserRole).value<QString>().endsWith("_annotation")) {
      QtAnnotation* annotation = _qtAnnotations[(*itm)->data(1, Qt::UserRole).value<QString>()];
      annotation->setSelected(true);
      _selectedAnnotations.insert(annotation);
      _activeAnnotation = annotation;
    }
    else {
      if ((*itm)->childCount() > 0) {
        QTreeWidgetItemIterator subItm((*itm)->child(0));
        while (*subItm && (*subItm)->parent() != (*itm)->parent()) {
          if ((*subItm)->data(1, Qt::UserRole).value<QString>().endsWith("_annotation")) {
            QtAnnotation* annotation = _qtAnnotations[(*subItm)->data(1, Qt::UserRole).value<QString>()];
            annotation->setSelected(true);
            _selectedAnnotations.insert(annotation);
            _activeAnnotation = annotation;
          }
          ++subItm;
        }
      }
    }
  }
}

void AnnotationWorkstationExtensionPlugin::onLoadButtonPressed(const std::string& filePath) {
  QString fileName;
  if (filePath.empty()) {
    fileName = QFileDialog::getOpenFileName(NULL, tr("Load annotations"), _settings->value("lastOpenendPath", QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)).toString(), tr("Annotation files(*.xml;*.ndpa)"));
  }
  else {
    fileName = QString::fromStdString(filePath);
  }
  if (!fileName.isEmpty()) {
    onClearButtonPressed();
    if (!_annotationService->loadRepositoryFromFile(fileName.toStdString())) {
      int ret = QMessageBox::warning(NULL, tr("ASAP"),
        tr("The annotations could not be loaded."),
        QMessageBox::Ok);
    }
    // Check if it is an ImageScopeRepository, if so, offer the user the chance to reload with new closing distance
    if (dynamic_cast<ImageScopeRepository*>(_annotationService->getRepository())) {
      bool ok = false;
      float newClosingDistance = QInputDialog::getDouble(_viewer, tr("Enter the annotation closing distance."), tr("Please provide the maximal distance for which annotations are automatically closed by ASAP if they remain open."), 30., 0, 1000, 1, &ok);
      float closingDistance = dynamic_cast<ImageScopeRepository*>(_annotationService->getRepository())->getClosingDistance();
      if (ok && newClosingDistance != closingDistance) {
        _annotationService->getList()->removeAllAnnotations();
        _annotationService->getList()->removeAllGroups();
        dynamic_cast<ImageScopeRepository*>(_annotationService->getRepository())->setClosingDistance(newClosingDistance);
        dynamic_cast<ImageScopeRepository*>(_annotationService->getRepository())->load();
      }
    }
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
        newAnnotationGroup->setText(1, QString::fromStdString((*it)->getName()));
        newAnnotationGroup->setText(2, "Group");
        newAnnotationGroup->setData(1, Qt::UserRole, QString::fromStdString(key));
        newAnnotationGroup->setFlags(newAnnotationGroup->flags() | Qt::ItemIsEditable);
        int cHeight = _treeWidget->visualItemRect(newAnnotationGroup).height();
        QPixmap iconPM(cHeight, cHeight);
        iconPM.fill(QColor((*it)->getColor().c_str()));
        QIcon color(iconPM);
        newAnnotationGroup->setIcon(0, color);
        newAnnotationGroup->setData(0, Qt::UserRole, QColor((*it)->getColor().c_str()));
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
          newAnnotationGroup->setText(1, QString::fromStdString((*it).second->getName()));
          newAnnotationGroup->setText(2, "Group");
          newAnnotationGroup->setData(1, Qt::UserRole, QString::fromStdString(key));
          newAnnotationGroup->setFlags(newAnnotationGroup->flags() | Qt::ItemIsEditable);
          int cHeight = _treeWidget->visualItemRect(newAnnotationGroup).height();
          QPixmap iconPM(cHeight, cHeight);
          iconPM.fill(QColor((*it).second->getColor().c_str()));
          QIcon color(iconPM);
          newAnnotationGroup->setIcon(0, color);
          newAnnotationGroup->setData(0, Qt::UserRole, QColor((*it).second->getColor().c_str()));
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
        _viewer->scene()->addItem(annot);
        annot->setZValue(20.);
      }

      _annotationIndex += 1;
      QTreeWidgetItem* newAnnotation = new QTreeWidgetItem(prnt);
      newAnnotation->setText(1, QString::fromStdString((*it)->getName()));
      newAnnotation->setText(2, QString::fromStdString((*it)->getTypeAsString()));
      newAnnotation->setFlags(newAnnotation->flags() & ~Qt::ItemIsDropEnabled);
      newAnnotation->setFlags(newAnnotation->flags() | Qt::ItemIsEditable);
      newAnnotation->setData(1, Qt::UserRole, QString::fromStdString(key));
      int cHeight = _treeWidget->visualItemRect(newAnnotation).height();
      if (_treeWidget->topLevelItemCount() > 0) {
        cHeight = _treeWidget->visualItemRect(_treeWidget->topLevelItem(0)).height();
      }
      QPixmap iconPM(cHeight, cHeight);
      iconPM.fill(QColor((*it)->getColor().c_str()));
      QIcon color(iconPM);
      newAnnotation->setIcon(0, color);
      newAnnotation->setData(0, Qt::UserRole, QColor((*it)->getColor().c_str()));

    }
    _treeWidget->resizeColumnToContents(0);
    _treeWidget->resizeColumnToContents(1);
  }
}

void AnnotationWorkstationExtensionPlugin::onSaveButtonPressed() {
  QDir defaultName = _settings->value("lastOpenendPath", QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)).toString();
  QString basename = QFileInfo(_settings->value("currentFile", QString()).toString()).completeBaseName();
  if (basename.isEmpty()) {
    basename = QString("annotation.xml");
  }
  else {
    basename += QString(".xml");
  }
  QString fileName = QFileDialog::getSaveFileName(NULL, tr("Save annotations"), defaultName.filePath(basename), tr("XML file (*.xml);TIF file (*.tif)"));
  if (_img && fileName.endsWith(".tif")) {
    std::vector<AnnotationGroup*> grps = this->_annotationService->getList()->getGroups();
    QDialog* nameToLabel = new QDialog();
    nameToLabel->setWindowTitle("Assign labels to annotation groups");
    QVBoxLayout* dialogLayout = new QVBoxLayout();
    QFormLayout* nameToLabelLayout = new QFormLayout();
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    if (grps.empty()) {
      QSpinBox* label = new QSpinBox();
      label->setMinimum(0);
      label->setValue(1);
      label->setObjectName("All annotations");
      nameToLabelLayout->addRow("All annotations", label);
    }
    else {
      for (unsigned int i = 0; i < grps.size(); ++i) {
        if (!grps[i]->getGroup()) {
          QSpinBox* label = new QSpinBox();
          QString grpName = QString::fromStdString(grps[i]->getName());
          label->setObjectName(grpName);
          label->setMinimum(0);
          label->setValue(i + 1);
          nameToLabelLayout->addRow(grpName, label);
        }
      }
    }
    dialogLayout->addLayout(nameToLabelLayout);
    QPushButton* cancel = new QPushButton("Cancel");
    QPushButton* ok = new QPushButton("Ok");
    cancel->setDefault(true);
    connect(cancel, SIGNAL(clicked()), nameToLabel, SLOT(reject()));
    connect(ok, SIGNAL(clicked()), nameToLabel, SLOT(accept()));
    buttonLayout->addWidget(cancel);
    buttonLayout->addWidget(ok);
    dialogLayout->addLayout(buttonLayout);
    nameToLabel->setLayout(dialogLayout);
    int rval = nameToLabel->exec();
    if (rval) {
      QList<QSpinBox*> assignedLabels = nameToLabel->findChildren<QSpinBox*>();
      std::map<std::string, int> nameToLab;
      for (QList<QSpinBox*>::iterator it = assignedLabels.begin(); it != assignedLabels.end(); ++it) {
        if ((*it)->objectName().toStdString() == "All annotations") {
          continue;
        }
        nameToLab[(*it)->objectName().toStdString()] = (*it)->value();
      }
      AnnotationToMask maskConverter;
      QtProgressMonitor monitor;
      maskConverter.setProgressMonitor(&monitor);
      QProgressDialog progressDialog;
      QObject::connect(&monitor, SIGNAL(progressChanged(int)), &progressDialog, SLOT(setValue(int)));
      progressDialog.setMinimum(0);
      progressDialog.setMaximum(100);
      progressDialog.setCancelButton(NULL);
      progressDialog.setWindowModality(Qt::WindowModal);
      progressDialog.setValue(0);
      progressDialog.show();
      QApplication::processEvents();
      maskConverter.convert(_annotationService->getList(), fileName.toStdString(), _img->getDimensions(), _img->getSpacing(), nameToLab);
      delete nameToLabel;
    }
  }
  else if (!fileName.isEmpty()) {
    if (!_annotationService->saveRepositoryToFile(fileName.toStdString())) {
      int ret = QMessageBox::warning(NULL, tr("ASAP"),
        tr("The annotations could not be saved."),
        QMessageBox::Ok);
    }
  }
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
        QString UID = (*it)->data(1, Qt::UserRole).value<QString>();
        if (!(*it)->parent()) {
          if (UID.endsWith("_group")) {
            _qtAnnotationGroups[UID]->setGroup(NULL);
          }
          else {
            _qtAnnotations[UID]->getAnnotation()->setGroup(NULL);
          }
        }
        else {
          QString parentUID = (*it)->parent()->data(1, Qt::UserRole).value<QString>();
          QString UID = (*it)->data(1, Qt::UserRole).value<QString>();
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
      _treeWidget->resizeColumnToContents(0);
      _treeWidget->resizeColumnToContents(1);
    }
  }
  else if (qobject_cast<QWidget*>(watched) == _treeWidget && event->type() == QEvent::KeyPress) {
    QKeyEvent* kpEvent = dynamic_cast<QKeyEvent*>(event);
    if (kpEvent->key() == Qt::Key::Key_Delete) {
      QList<QTreeWidgetItem*> selItems = _treeWidget->selectedItems();
      // Handle selected items iteratively to make sure we do not accidentely remove the parent before the child
      while (!selItems.empty()) {
        QTreeWidgetItem* itm = selItems[0];
        if (itm->data(1, Qt::UserRole).value<QString>().endsWith("_group")) {
          deleteAnnotationGroup(_qtAnnotationGroups[itm->data(1, Qt::UserRole).value<QString>()]);
        }
        else {
          deleteAnnotation(_qtAnnotations[itm->data(1, Qt::UserRole).value<QString>()]);
        }
        selItems = _treeWidget->selectedItems();
      }
      connect(_treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(onTreeWidgetSelectedItemsChanged()));
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
    newAnnotationGroup->setText(1, QString::fromStdString(grp->getName()));
    newAnnotationGroup->setText(2, "Group");
    newAnnotationGroup->setData(1, Qt::UserRole, grpUID);
    newAnnotationGroup->setFlags(newAnnotationGroup->flags() | Qt::ItemIsEditable);
    int cHeight = _treeWidget->visualItemRect(newAnnotationGroup).height();
    QPixmap iconPM(cHeight, cHeight);
    iconPM.fill(QColor("#64FE2E"));
    QIcon color(iconPM);
    newAnnotationGroup->setIcon(0, color);
    newAnnotationGroup->setData(0, Qt::UserRole, QColor("#64FE2E"));
    _treeWidget->resizeColumnToContents(0);
    _treeWidget->resizeColumnToContents(1);
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
    if (core::fileExists(annotationPath)) {
      onLoadButtonPressed(annotationPath);
    }
  }
  _img = img;
}

void AnnotationWorkstationExtensionPlugin::onImageClosed() {
  if (_dockWidget) {
    _dockWidget->setEnabled(false);
  }
  onClearButtonPressed();
  _img = NULL;
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
    _treeWidget->clearSelection();
    _viewer->scene()->addItem(_generatedAnnotation);
    _generatedAnnotation->setZValue(20.);
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
      newAnnotation->setText(1, QString::fromStdString(_generatedAnnotation->getAnnotation()->getName()));
      newAnnotation->setText(2, QString::fromStdString(_generatedAnnotation->getAnnotation()->getTypeAsString()));
      newAnnotation->setFlags(newAnnotation->flags() & ~Qt::ItemIsDropEnabled);
      newAnnotation->setFlags(newAnnotation->flags() | Qt::ItemIsEditable);
      newAnnotation->setData(1, Qt::UserRole, annotUID);
      newAnnotation->setSelected(true);
      int cHeight = _treeWidget->visualItemRect(newAnnotation).height();
      QPixmap iconPM(cHeight, cHeight);
      iconPM.fill(QColor("yellow"));
      QIcon color(iconPM);
      newAnnotation->setIcon(0, color);
      newAnnotation->setData(0, Qt::UserRole, QColor("yellow"));
      _treeWidget->resizeColumnToContents(0);      
      _treeWidget->resizeColumnToContents(1);
      _activeAnnotation = _generatedAnnotation;
      _generatedAnnotation = NULL;
    }
    else {
      _viewer->scene()->removeItem(_generatedAnnotation);
      delete _generatedAnnotation->getAnnotation();
      _generatedAnnotation->deleteLater();
      _generatedAnnotation = NULL;
    }
  }
}

void AnnotationWorkstationExtensionPlugin::deleteAnnotation(QtAnnotation* annotation) {
  if (annotation) {
    if (_treeWidget) {
      QTreeWidgetItemIterator it(_treeWidget);
      while (*it) {
        if ((*it)->text(1) == QString::fromStdString(annotation->getAnnotation()->getName())) {
          QString annotUID = (*it)->data(1, Qt::UserRole).value<QString>();
          _qtAnnotations.remove(annotUID);
          if (_viewer) {
            _viewer->scene()->removeItem(annotation);
          }
          if (_annotationService) {
            _annotationService->getList()->removeAnnotation(annotation->getAnnotation()->getName());
          }
          annotation->deleteLater();
          _selectedAnnotations.remove(annotation);
          (*it)->setSelected(false);
          delete (*it);
          break;
        }
        ++it;
      }
    }
  }
}

void AnnotationWorkstationExtensionPlugin::deleteAnnotationGroup(AnnotationGroup* group) {
  if (_treeWidget) {
    QTreeWidgetItemIterator it(_treeWidget);
    while (*it) {
      if ((*it)->text(1) == QString::fromStdString(group->getName())) {
        if ((*it)->childCount() > 0) {
          for (int i = (*it)->childCount() - 1; i >= 0; --i) {
            QTreeWidgetItem* itm = (*it)->child(i);
            if (itm->data(1, Qt::UserRole).value<QString>().endsWith("_group")) {
              deleteAnnotationGroup(_qtAnnotationGroups[itm->data(1, Qt::UserRole).value<QString>()]);
            }
            else {
              deleteAnnotation(_qtAnnotations[itm->data(1, Qt::UserRole).value<QString>()]);
            }
          }
        }
        QString groupUID = (*it)->data(1, Qt::UserRole).value<QString>();
        if (_annotationService) {
          _annotationService->getList()->removeGroup(_qtAnnotationGroups[groupUID]->getName());
        }
        _qtAnnotationGroups.remove(groupUID);
        (*it)->setSelected(false);
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

void AnnotationWorkstationExtensionPlugin::clearSelection() {
  if (_treeWidget) {
    _treeWidget->clearSelection();
  }
}

void AnnotationWorkstationExtensionPlugin::addAnnotationToSelection(QtAnnotation* annotation) {
  QTreeWidgetItemIterator it(_treeWidget);
  while (*it) {
    if ((*it)->text(1) == QString::fromStdString(annotation->getAnnotation()->getName())) {
      (*it)->setSelected(true);
      break;
    }
    ++it;
  }
}

void AnnotationWorkstationExtensionPlugin::removeAnnotationFromSelection(QtAnnotation* annotation) {
  QTreeWidgetItemIterator it(_treeWidget);
  while (*it) {
    if ((*it)->text(1) == QString::fromStdString(annotation->getAnnotation()->getName())) {
      (*it)->setSelected(true);
      break;
    }
    ++it;
  }
}

QSet<QtAnnotation*> AnnotationWorkstationExtensionPlugin::getSelectedAnnotations() {
  return _selectedAnnotations;
}