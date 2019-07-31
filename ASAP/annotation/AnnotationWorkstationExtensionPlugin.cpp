#include "AnnotationWorkstationExtensionPlugin.h"
#include "DotAnnotationTool.h"
#include "PolyAnnotationTool.h"
#include "PointSetAnnotationTool.h"
#include "SplineAnnotationTool.h"
#include "annotation/AnnotationService.h"
#include "annotation/AnnotationList.h"
#include "annotation/AnnotationGroup.h"
#include "QtAnnotation.h"
#include "QtAnnotationGroup.h"
#include "annotation/Annotation.h"
#include "annotation/ImageScopeRepository.h"
#include "annotation/AnnotationToMask.h"
#include "DotQtAnnotation.h"
#include "PolyQtAnnotation.h"
#include "MeasurementQtAnnotation.h"
#include "RectangleAnnotationTool.h"
#include "MeasurementAnnotationTool.h"
#include "PointSetQtAnnotation.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
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
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QFormLayout>
#include <QSpinBox>
#include "core/filetools.h"
#include "../QtProgressMonitor.h"
#include <QProgressDialog>
#include <QMessageBox>
#include <QInputDialog>

#include <numeric>
#include <iostream>

AnnotationPluginInfo::AnnotationPluginInfo(void)
	: annotation_index(0), annotation_group_index(0), annotation_service(new AnnotationService()), initial_file_loaded(false)
{
}

AnnotationWorkstationExtensionPlugin::AnnotationWorkstationExtensionPlugin() :
  WorkstationExtensionPluginInterface(),
  _generatedAnnotation(NULL),
  _activeAnnotation(NULL),
  _dockWidget(NULL),
  _treeWidget(NULL),
  _oldEvent(NULL),
  _currentAnnotationLine(NULL),
  _currentAnnotationLabel(NULL),
  _currentAnnotationHeaderLabel(NULL),
  _currentPixelArea(1.),
  m_annotation_items_(nullptr),
  m_viewer_(nullptr),
  m_instance_(nullptr),
  m_instance_information_(nullptr)
{
  QUiLoader loader;
  QFile file(":/AnnotationWorkstationExtensionPlugin_ui/AnnotationDockWidget.ui");
  bool openend = file.open(QFile::ReadOnly);
  _dockWidget = qobject_cast<QDockWidget*>(loader.load(&file));
  if (_dockWidget) {
    _dockWidget->setEnabled(false);
    _treeWidget = _dockWidget->findChild<QTreeWidget*>("AnnotationTreeWidget");
    _treeWidget->viewport()->installEventFilter(this);
    _treeWidget->setMouseTracking(true);
    _treeWidget->installEventFilter(this);
    QPushButton* groupButton = _dockWidget->findChild<QPushButton*>("addGroupButton");
    QPushButton* clearButton = _dockWidget->findChild<QPushButton*>("clearButton");
    QPushButton* saveButton = _dockWidget->findChild<QPushButton*>("saveButton");
    QPushButton* loadButton = _dockWidget->findChild<QPushButton*>("loadButton");
    QPushButton* optionsButton = _dockWidget->findChild<QPushButton*>("optionsButton");
    _currentAnnotationLine = _dockWidget->findChild<QFrame*>("currentAnnotationLine");
    _currentAnnotationLabel = _dockWidget->findChild<QLabel*>("currentAnnotationLabel");
    _currentAnnotationHeaderLabel = _dockWidget->findChild<QLabel*>("currentAnnotationHeaderLabel");
    _currentAnnotationLine->setVisible(false);
    _currentAnnotationLabel->setVisible(false);
    _currentAnnotationHeaderLabel->setVisible(false);
    connect(groupButton, SIGNAL(clicked()), this, SLOT(addAnnotationGroup()));
    connect(clearButton, SIGNAL(clicked()), this, SLOT(onClearButtonPressed()));
    connect(saveButton, SIGNAL(clicked()), this, SLOT(onSaveButtonPressed()));
    connect(loadButton, SIGNAL(clicked()), this, SLOT(onLoadButtonPressed()));
    connect(optionsButton, SIGNAL(clicked()), this, SLOT(onOptionsButtonPressed()));
    connect(_treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(onItemNameChanged(QTreeWidgetItem*, int)));
    connect(_treeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(onTreeWidgetItemDoubleClicked(QTreeWidgetItem*, int)));
    connect(_treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(onTreeWidgetSelectedItemsChanged()));
    connect(_treeWidget, SIGNAL(itemExpanded(QTreeWidgetItem*)), this, SLOT(resizeOnExpand()));
  }
  _settings = new QSettings(QSettings::IniFormat, QSettings::UserScope, "DIAG", "ASAP", this);

  for (int i = 0; i < QColorDialog::customCount(); ++i) {
    QColor customColor = _settings->value("annotationCustomColor" + QString(i), QColor("white")).value<QColor>();
    QColorDialog::setCustomColor(i, customColor);
  }
  QtAnnotation::selectionSensitivity = _settings->value("annotationSelectionSensitivity", 100.).value<float>();

  qRegisterMetaTypeStreamOperators<QtAnnotation*>("QtAnnotation*");
  qRegisterMetaTypeStreamOperators<QtAnnotationGroup*>("QtAnnotationGroup*");
}

AnnotationWorkstationExtensionPlugin::~AnnotationWorkstationExtensionPlugin() {
  for (int i = 0; i < QColorDialog::customCount(); ++i) {
    QColor customColor = QColorDialog::customColor(i);
    _settings->setValue("annotationCustomColor" + QString(i), customColor);
  }
  clear(false, true);
  m_mutex_.lock();
  m_mutex_.unlock();
}

void AnnotationWorkstationExtensionPlugin::onClearButtonPressed() {
  if (shouldClear(m_instance_information_)) {
    clear(true, true);
  }
}

void AnnotationWorkstationExtensionPlugin::onOptionsButtonPressed() {
  QDialog* optionsDialog = new QDialog();
  optionsDialog->setWindowTitle("Set options for annotation tools");
  QVBoxLayout* dialogLayout = new QVBoxLayout();
  QFormLayout* optionsDialogLayout = new QFormLayout();
  QHBoxLayout* buttonLayout = new QHBoxLayout();
  QDoubleSpinBox * selSensSpinBox = new QDoubleSpinBox();
  selSensSpinBox->setMinimum(20);
  selSensSpinBox->setMaximum(1000);
  selSensSpinBox->setValue(QtAnnotation::selectionSensitivity);
  selSensSpinBox->setSingleStep(20);
  selSensSpinBox->setObjectName("SelectionSensitivity");
  selSensSpinBox->setToolTip("Sets the selection sensitivy for clicking individual points in an annatation. Higher means easier to click.");
  optionsDialogLayout->addRow("Selection sensitivity", selSensSpinBox);
  dialogLayout->addLayout(optionsDialogLayout);
  QPushButton* cancel = new QPushButton("Cancel");
  QPushButton* ok = new QPushButton("Ok");
  cancel->setDefault(true);
  connect(cancel, SIGNAL(clicked()), optionsDialog, SLOT(reject()));
  connect(ok, SIGNAL(clicked()), optionsDialog, SLOT(accept()));
  buttonLayout->addWidget(cancel);
  buttonLayout->addWidget(ok);
  dialogLayout->addLayout(buttonLayout);
  optionsDialog->setLayout(dialogLayout);
  int rval = optionsDialog->exec();
  if (rval) {
    float newSelectionSensitivity = static_cast<float>(selSensSpinBox->value());
    QtAnnotation::selectionSensitivity = newSelectionSensitivity;
    _settings->setValue("annotationSelectionSensitivity", newSelectionSensitivity);
  }
}

void AnnotationWorkstationExtensionPlugin::clear(const bool clear_instance_list, const bool delete_from_scene) {
	m_mutex_.lock();
  if (_generatedAnnotation) {
    for (std::vector<std::shared_ptr<ToolPluginInterface> >::iterator it = _annotationTools.begin(); it != _annotationTools.end(); ++it) {
      std::dynamic_pointer_cast<AnnotationTool>((*it))->cancelAnnotation();
    }
  }
  _treeWidget->clearSelection();
  clearTreeWidget();
  clearQtAnnotations(delete_from_scene);

  if (clear_instance_list)
  {
	  clearAnnotationList();
	  m_instance_information_->annotation_group_index = 0;
	  m_instance_information_->annotation_index = 0;
  }
  m_mutex_.unlock();
}

void AnnotationWorkstationExtensionPlugin::resizeOnExpand() {
  if (_treeWidget) {
    _treeWidget->resizeColumnToContents(0);
    _treeWidget->resizeColumnToContents(1);
  }
}

void AnnotationWorkstationExtensionPlugin::clearTreeWidget() {
  if (_treeWidget) {
    _annotToItem.clear();
    _treeWidget->clear();
  }
}

void AnnotationWorkstationExtensionPlugin::clearAnnotationList() {
	  m_instance_information_->annotation_service->getList()->removeAllAnnotations();
	  m_instance_information_->annotation_service->getList()->removeAllGroups();
}

void AnnotationWorkstationExtensionPlugin::clearQtAnnotations(const bool delete_from_scene) {
	if (m_annotation_items_)
	{
		if (delete_from_scene)
		{
			for (QtAnnotation* annotation : m_annotation_items_->annotations)
			{
				m_viewer_->scene()->removeItem(annotation);
				annotation->deleteLater();
			}
		}
		m_annotation_items_->annotations.clear();
	}
  _activeAnnotation = NULL;
  _generatedAnnotation = NULL;
}

void AnnotationWorkstationExtensionPlugin::onItemNameChanged(QTreeWidgetItem* item, int column) {
  if (item && column == 1) {
    if (QtAnnotation* annot = item->data(1, Qt::UserRole).value<QtAnnotation*>()) {
       annot->getAnnotation()->setName(item->text(1).toStdString());
    }
    else {
      QtAnnotationGroup* grp = item->data(1, Qt::UserRole).value<QtAnnotationGroup*>();
      if (grp) {
        grp->getAnnotationGroup()->setName(item->text(1).toStdString());
      }
    }
  }
}

void AnnotationWorkstationExtensionPlugin::updateAnnotationToolTip(QtAnnotation* annotation) {
  if (annotation) {
    QMap<QtAnnotation*, QTreeWidgetItem*>::iterator it = _annotToItem.find(annotation);
    if (it != _annotToItem.end()) {
      unsigned int nrPoints = annotation->getAnnotation()->getNumberOfPoints();
      float area = annotation->getAnnotation()->getArea();
      QString areaUnit(" pixels.");
      if (_currentPixelArea != 1.) {
        areaUnit = QString(" um<sup>2</sup></html>");
      }
      it.value()->setToolTip(1, QString("<html>Total number of control points: ") + QString::number(nrPoints) + QString("<br/>") + QString("Total area: ") + QString::number(area * _currentPixelArea, 'g', 4) + areaUnit);
      QTreeWidgetItem* parentItem = it.value()->parent();
      while (parentItem) {
        if (QtAnnotationGroup* grp = parentItem->data(1, Qt::UserRole).value<QtAnnotationGroup*>()) {
          unsigned int nrPoints = grp->getAnnotationGroup()->getNumberOfPoints();
          float area = grp->getAnnotationGroup()->getArea();
          QString areaUnit(" pixels.");
          if (_currentPixelArea != 1.) {
            areaUnit = QString(" um<sup>2</sup></html>");
          }
          parentItem->setToolTip(1, QString("<html>Total number of control points: ") + QString::number(nrPoints) + QString("<br/>") + QString("Total area: ") + QString::number(area * _currentPixelArea, 'g', 4) + areaUnit);
        }
        else if (QtAnnotation* annot = parentItem->data(1, Qt::UserRole).value<QtAnnotation*>()) {
          unsigned int nrPoints = annot->getAnnotation()->getNumberOfPoints();
          float area = annot->getAnnotation()->getArea();
          QString areaUnit(" pixels.");
          if (_currentPixelArea != 1.) {
            areaUnit = QString(" um<sup>2</sup></html>");
          }
          parentItem->setToolTip(1, QString("<html>Total number of control points: ") + QString::number(nrPoints) + QString("<br/>") + QString("Total area: ") + QString::number(area * _currentPixelArea, 'g', 4) + areaUnit);
        }
        parentItem = parentItem->parent();
      }
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
      if (QtAnnotation* annot = item->data(1, Qt::UserRole).value<QtAnnotation*>()) {
        annot->getAnnotation()->setColor(newColor.name().toStdString());
      }
      else {
        QtAnnotationGroup* grp = item->data(1, Qt::UserRole).value<QtAnnotationGroup* >();
        if (grp) {
            grp->getAnnotationGroup()->setColor(newColor.name().toStdString());
        }
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
    if (QtAnnotation* annot = (*itm)->data(1, Qt::UserRole).value<QtAnnotation*>()) {
      annot->setSelected(true);
      _selectedAnnotations.insert(annot);
      _activeAnnotation = annot;
    }
    else {
      if ((*itm)->childCount() > 0) {
        QTreeWidgetItemIterator subItm((*itm)->child(0));
        while (*subItm && (*subItm)->parent() != (*itm)->parent()) {
          if (QtAnnotation* annot = (*itm)->data(1, Qt::UserRole).value<QtAnnotation*>()) {
            annot->setSelected(true);
            _selectedAnnotations.insert(annot);
            _activeAnnotation = annot;
          }
          ++subItm;
        }
      }
    }
  }
}

bool AnnotationWorkstationExtensionPlugin::canClose(ASAP::DocumentInstance& instance) {
  return shouldClear((AnnotationPluginInfo*)instance.GetPluginInformation("annotation"));
}

bool AnnotationWorkstationExtensionPlugin::shouldClear(AnnotationPluginInfo* info) {
	bool shouldClear = false;
	if (info)
	{
		if (info->annotation_service->getList()->isModified()) {
			QMessageBox msgBox;
			msgBox.setText("The annotations have been modified.");
			msgBox.setInformativeText("Do you want to save your changes?");
			msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
			msgBox.setDefaultButton(QMessageBox::Save);
			int ret = msgBox.exec();
			switch (ret) {
			case QMessageBox::Save:
				shouldClear = onSaveButtonPressed();
				break;
			case QMessageBox::Discard:
				shouldClear = true;
				break;
			case QMessageBox::Cancel:
				shouldClear = false;
			default:
				shouldClear = false;
			}
		}
		else {
			shouldClear = true;
		}
	}
	else
	{
		shouldClear = true;
	}
  return shouldClear;
}

void AnnotationWorkstationExtensionPlugin::onLoadButtonPressed(const std::string& filePath) {
  if (m_instance_)
  {
	  QString fileName;
	  if (filePath.empty()) {
		  fileName = QFileDialog::getOpenFileName(NULL, tr("Load annotations"), _settings->value("lastOpenendPath", QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)).toString(), tr("Annotation files (*.xml *.ndpa)"));
	  }
	  else {
		  fileName = QString::fromStdString(filePath);
	  }
	  if (!fileName.isEmpty())
	  {
		  if (!shouldClear(m_instance_information_)) {
			  return;
		  }
		  clear(false, true);
		  if (!m_instance_information_->annotation_service->loadRepositoryFromFile(fileName.toStdString())) {
			  int ret = QMessageBox::warning(NULL, tr("ASAP"),
				  tr("The annotations could not be loaded."),
				  QMessageBox::Ok);
		  }

		  // Check if it is an ImageScopeRepository, if so, offer the user the chance to reload with new closing distance
		  std::shared_ptr<ImageScopeRepository> imscRepo = std::dynamic_pointer_cast<ImageScopeRepository>(m_instance_information_->annotation_service->getRepository());
		  if (imscRepo) {
			  bool ok = false;
			  bool ok2 = false;
			  float newClosingDistance = QInputDialog::getDouble(m_viewer_, tr("Enter the annotation closing distance."), tr("Please provide the maximal distance for which annotations are automatically closed by ASAP if they remain open."), 30., 0, 1000, 1, &ok);
			  float simplifcationDistance = QInputDialog::getDouble(m_viewer_, tr("Enter the simplification distance."), tr("Please provide the minimal perpendicular distance for which points are automatically merged."), 100, 0, 1000, 2, &ok2);
			  float closingDistance = imscRepo->getClosingDistance();
			  if (ok && newClosingDistance != closingDistance) {
				  m_instance_information_->annotation_service->getList()->removeAllAnnotations();
				  m_instance_information_->annotation_service->getList()->removeAllGroups();
				  imscRepo->setClosingDistance(newClosingDistance);
				  imscRepo->load();
			  }
			  if (ok2 && simplifcationDistance > 0) {
				  std::vector<std::shared_ptr<Annotation> > annotations = m_instance_information_->annotation_service->getList()->getAnnotations();
				  for (std::vector<std::shared_ptr<Annotation> >::iterator it = annotations.begin(); it != annotations.end(); ++it) {
					  (*it)->simplify(0, simplifcationDistance);
				  }
			  }
		  }
		  // Add loaded groups to treewidget
		  QList<QtAnnotationGroup* > childGroups;
		  std::map<std::shared_ptr<AnnotationGroup>, QTreeWidgetItem*> annotToWidget;
		  std::vector<std::shared_ptr<AnnotationGroup> > grps = m_instance_information_->annotation_service->getList()->getGroups();
		  for (std::vector<std::shared_ptr<AnnotationGroup> >::const_iterator it = grps.begin(); it != grps.end(); ++it) {
			  QtAnnotationGroup *grp = new QtAnnotationGroup(*it, this);
			  if ((*it)->getGroup() == NULL) {
				  m_annotation_items_->groups.append(grp);
				  QTreeWidgetItem* newAnnotationGroup = new QTreeWidgetItem(_treeWidget);
				  newAnnotationGroup->setText(1, QString::fromStdString((*it)->getName()));
				  newAnnotationGroup->setText(2, "Group");
				  newAnnotationGroup->setData(1, Qt::UserRole, QVariant::fromValue<QtAnnotationGroup*>(grp));
				  newAnnotationGroup->setFlags(newAnnotationGroup->flags() | Qt::ItemIsEditable);
				  int cHeight = _treeWidget->visualItemRect(newAnnotationGroup).height();
				  QPixmap iconPM(cHeight, cHeight);
				  iconPM.fill(QColor((*it)->getColor().c_str()));
				  QIcon color(iconPM);
				  newAnnotationGroup->setIcon(0, color);
				  newAnnotationGroup->setData(0, Qt::UserRole, QColor((*it)->getColor().c_str()));
				  annotToWidget[grp->getAnnotationGroup()] = newAnnotationGroup;
			  }
			  else {
				  childGroups.append(grp);
			  }
		  }
		  while (!childGroups.empty()) {
			  for (QList<QtAnnotationGroup*>::iterator it = childGroups.begin(); it != childGroups.end();) {
				  if (annotToWidget.find((*it)->getAnnotationGroup()->getGroup()) != annotToWidget.end()) {
					  m_annotation_items_->groups.append((*it));
					  QTreeWidgetItem* newAnnotationGroup = new QTreeWidgetItem(annotToWidget[(*it)->getAnnotationGroup()->getGroup()]);
					  newAnnotationGroup->setText(1, QString::fromStdString((*it)->getAnnotationGroup()->getName()));
					  newAnnotationGroup->setText(2, "Group");
					  newAnnotationGroup->setData(1, Qt::UserRole, QVariant::fromValue<QtAnnotationGroup*>((*it)));
					  newAnnotationGroup->setFlags(newAnnotationGroup->flags() | Qt::ItemIsEditable);
					  int cHeight = _treeWidget->visualItemRect(newAnnotationGroup).height();
					  QPixmap iconPM(cHeight, cHeight);
					  iconPM.fill(QColor((*it)->getAnnotationGroup()->getColor().c_str()));
					  QIcon color(iconPM);
					  newAnnotationGroup->setIcon(0, color);
					  newAnnotationGroup->setData(0, Qt::UserRole, QColor((*it)->getAnnotationGroup()->getColor().c_str()));
					  annotToWidget[(*it)->getAnnotationGroup()] = newAnnotationGroup;
					  it = childGroups.erase(it);
				  }
				  else {
					  ++it;
				  }
			  }
		  }
		  std::vector<std::shared_ptr<Annotation> > annots = m_instance_information_->annotation_service->getList()->getAnnotations();
		  for (std::vector<std::shared_ptr<Annotation> >::const_iterator it = annots.begin(); it != annots.end(); ++it) {
			  QTreeWidgetItem* prnt = _treeWidget->invisibleRootItem();
			  if ((*it)->getGroup()) {
				  prnt = annotToWidget[(*it)->getGroup()];
			  }
			  std::string key = "Annotation " + QString::number(m_instance_information_->annotation_index).toStdString() + "_annotation";

			  // Add QtAnnotation
			  QtAnnotation* annot = NULL;
			  if ((*it)->getType() == Annotation::Type::DOT) {
				  annot = new DotQtAnnotation((*it), this, m_viewer_->getSceneScale());
			  }
			  else if ((*it)->getType() == Annotation::Type::POLYGON || (*it)->getType() == Annotation::Type::RECTANGLE) {
				  annot = new PolyQtAnnotation((*it), this, m_viewer_->getSceneScale());
				  dynamic_cast<PolyQtAnnotation*>(annot)->setInterpolationType("linear");
			  }
			  else if ((*it)->getType() == Annotation::Type::SPLINE) {
				  annot = new PolyQtAnnotation((*it), this, m_viewer_->getSceneScale());
				  dynamic_cast<PolyQtAnnotation*>(annot)->setInterpolationType("spline");
			  }
			  else if ((*it)->getType() == Annotation::Type::MEASUREMENT) {
				  annot = new MeasurementQtAnnotation((*it), this, m_viewer_->getSceneScale());
			  }
			  else if ((*it)->getType() == Annotation::Type::POINTSET) {
				  annot = new PointSetQtAnnotation((*it), this, m_viewer_->getSceneScale());
			  }
			  else if ((*it)->getType() == Annotation::Type::RECTANGLE) {
				  annot = new PolyQtAnnotation((*it), this, m_viewer_->getSceneScale());
			  }
			  if (annot) {
				  annot->finish();
				  m_annotation_items_->annotations.append(annot);
				  m_viewer_->scene()->addItem(annot);
				  annot->setZValue(20.);

				  m_instance_information_->annotation_index += 1;
				  QTreeWidgetItem* newAnnotation = new QTreeWidgetItem(prnt);
				  newAnnotation->setText(1, QString::fromStdString((*it)->getName()));
				  newAnnotation->setText(2, QString::fromStdString((*it)->getTypeAsString()));
				  newAnnotation->setFlags(newAnnotation->flags() & ~Qt::ItemIsDropEnabled);
				  newAnnotation->setFlags(newAnnotation->flags() | Qt::ItemIsEditable);
				  newAnnotation->setData(1, Qt::UserRole, QVariant::fromValue<QtAnnotation*>(annot));
				  int cHeight = _treeWidget->visualItemRect(newAnnotation).height();
				  if (_treeWidget->topLevelItemCount() > 0) {
					  cHeight = _treeWidget->visualItemRect(_treeWidget->topLevelItem(0)).height();
				  }
				  QPixmap iconPM(cHeight, cHeight);
				  iconPM.fill(QColor((*it)->getColor().c_str()));
				  QIcon color(iconPM);
				  newAnnotation->setIcon(0, color);
				  newAnnotation->setData(0, Qt::UserRole, QColor((*it)->getColor().c_str()));
				  _annotToItem[annot] = newAnnotation;
				  updateAnnotationToolTip(annot);
				  connect(annot, SIGNAL(annotationChanged(QtAnnotation*)), this, SLOT(updateAnnotationToolTip(QtAnnotation*)));
			  }
		  }
		  _treeWidget->resizeColumnToContents(0);
		  _treeWidget->resizeColumnToContents(1);
	  }
  }
}

bool AnnotationWorkstationExtensionPlugin::onSaveButtonPressed() {
  QDir defaultName = _settings->value("lastOpenendPath", QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)).toString();
  QString basename = QFileInfo(_settings->value("currentFile", QString()).toString()).completeBaseName();
  if (basename.isEmpty()) {
    basename = QString("annotation.xml");
  }
  else {
    basename += QString(".xml");
  }
  QString fileName = QFileDialog::getSaveFileName(NULL, tr("Save annotations"), defaultName.filePath(basename), tr("XML file (*.xml);;TIF file (*.tif);;All files (*)"));
  if (fileName.endsWith(".tif")) {
    if (std::shared_ptr<MultiResolutionImage> local_img = m_instance_->document->GetImage().lock()) {
      std::vector<std::shared_ptr<AnnotationGroup> > grps = m_instance_information_->annotation_service->getList()->getGroups();
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
        maskConverter.convert(m_instance_information_->annotation_service->getList(), fileName.toStdString(), local_img->getDimensions(), local_img->getSpacing(), nameToLab);
        delete nameToLabel;
        return true;
      }
    }
    else {
      return false;
    }
  }
  else if (!fileName.isEmpty()) {
    if (!m_instance_information_->annotation_service->saveRepositoryToFile(fileName.toStdString())) {
      int ret = QMessageBox::warning(NULL, tr("ASAP"),
        tr("The annotations could not be saved."),
        QMessageBox::Ok);
      return false;
    }
    else {
		m_instance_information_->annotation_service->getList()->resetModifiedStatus();
      return true;
    }
  }
  return false;
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
        QtAnnotation* annot = (*it)->data(1, Qt::UserRole).value<QtAnnotation*>();
        if (!(*it)->parent()) {
          if (annot) {            
            annot->getAnnotation()->setGroup(NULL);
          }
          else {
            QtAnnotationGroup* grp = (*it)->data(1, Qt::UserRole).value<QtAnnotationGroup*>();
            if (grp) {
              grp->getAnnotationGroup()->setGroup(NULL);
            }
          }
        }
        else {
          QtAnnotation* annot = (*it)->data(1, Qt::UserRole).value<QtAnnotation*>();
          if (annot) {
            annot->getAnnotation()->setGroup((*it)->parent()->data(1, Qt::UserRole).value<QtAnnotationGroup*>()->getAnnotationGroup());
          }
          else {
            QtAnnotationGroup* grp = (*it)->data(1, Qt::UserRole).value<QtAnnotationGroup*>();
            if (grp) {
              grp->getAnnotationGroup()->setGroup((*it)->parent()->data(1, Qt::UserRole).value<QtAnnotationGroup*>()->getAnnotationGroup());
            }
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
        if (QtAnnotationGroup* grp = itm->data(1, Qt::UserRole).value<QtAnnotationGroup*>()) {
          deleteAnnotationGroup(grp);
        }
        else {
          deleteAnnotation(itm->data(1, Qt::UserRole).value<QtAnnotation*>());
        }
        selItems = _treeWidget->selectedItems();
      }
      connect(_treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(onTreeWidgetSelectedItemsChanged()));
    }
    else if (kpEvent->key() == Qt::Key::Key_Z) {
      if (std::shared_ptr<AnnotationTool> tool = std::dynamic_pointer_cast<AnnotationTool>(_controller->GetActiveTool())) {
        tool->keyPressEvent(kpEvent);
      }
    }
  }
  return QObject::eventFilter(watched, event);
}

void AnnotationWorkstationExtensionPlugin::addAnnotationGroup() {
  if (_treeWidget && m_instance_information_) {
    std::shared_ptr<AnnotationGroup> grp = std::make_shared<AnnotationGroup>();
    QtAnnotationGroup* annotGroup = new QtAnnotationGroup(grp, this);
    grp->setName("Annotation Group " + QString::number(m_instance_information_->annotation_index).toStdString());
	m_instance_information_->annotation_group_index += 1;
    QString grpUID = QString::fromStdString(grp->getName() + "_group");
	m_instance_information_->annotation_service->getList()->addGroup(grp);

	addAnnotationGroupToGUI_(annotGroup);
  }
}

QDockWidget* AnnotationWorkstationExtensionPlugin::getDockWidget() {
  return _dockWidget;
}

void AnnotationWorkstationExtensionPlugin::onDocumentChange(ASAP::DocumentInstance* instance)
{
	clear(false, false);
	m_instance_			= instance;

	if (instance)
	{
		// Acquires or creates an entry into the current DocumentInstance
		if (!m_instance_->HasPluginInformation("annotation"))
		{
			m_instance_->SetPluginInformation("annotation", new AnnotationPluginInfo(), false);
		}
		m_instance_information_ = (AnnotationPluginInfo*)m_instance_->GetPluginInformation("annotation");

		if (_dockWidget) {
			_dockWidget->setEnabled(true);
		}

		std::string filename(instance->document->GetFilepath().string());
		if (!filename.empty()) {
			std::string annotationPath = filename;
			core::changeExtension(annotationPath, "xml");
			if (core::fileExists(annotationPath) && !m_instance_information_->initial_file_loaded) {
				onLoadButtonPressed(annotationPath);
				m_instance_information_->initial_file_loaded = true;
			}
		}
		if (std::shared_ptr<MultiResolutionImage> local_img = instance->document->GetImage().lock()) {
			std::vector<double> spacing = local_img->getSpacing();
			if (spacing.size() > 1) {
				_currentPixelArea = spacing[0] * spacing[1];
			}
			else {
				_currentPixelArea = 1.;
			}
		}

		// Inserts potential annotations into the GUI
		for (std::shared_ptr<Annotation>& annotation : m_instance_information_->annotation_service->getList()->getAnnotations())
		{
			m_annotation_items_->annotations.push_back(InitializeAnnotation_(annotation));
			addAnnotationToGUI_(m_annotation_items_->annotations.back());
		}
		for (std::shared_ptr<AnnotationGroup>& group : m_instance_information_->annotation_service->getList()->getGroups())
		{
			addAnnotationGroupToGUI_(new QtAnnotationGroup(group, this));
		}

		_treeWidget->clearSelection();
	}
	else if (_dockWidget)
	{
		_dockWidget->setEnabled(false);
		m_instance_information_ = nullptr;
	}
}

std::weak_ptr<MultiResolutionImage> AnnotationWorkstationExtensionPlugin::getCurrentImage() {
	if (m_instance_)
	{
		return m_instance_->document->GetImage();
	}
	return std::weak_ptr<MultiResolutionImage>();
}

bool AnnotationWorkstationExtensionPlugin::initialize(ASAP::PathologyViewController& controller) {
  WorkstationExtensionPluginInterface::initialize(controller);

  std::shared_ptr<ToolPluginInterface> tool(new DotAnnotationTool(this, controller));
  _annotationTools.push_back(tool);
  tool.reset(new RectangleAnnotationTool(this, controller));
  _annotationTools.push_back(tool);
  tool.reset(new PolyAnnotationTool(this, controller));
  _annotationTools.push_back(tool);
  tool.reset(new SplineAnnotationTool(this, controller));
  _annotationTools.push_back(tool);
  tool.reset(new PointSetAnnotationTool(this, controller));
  _annotationTools.push_back(tool);
  tool.reset(new MeasurementAnnotationTool(this, controller));
  _annotationTools.push_back(tool);
  return true;
}

std::vector<std::shared_ptr<ToolPluginInterface> > AnnotationWorkstationExtensionPlugin::getTools() {
  return _annotationTools;
}

void AnnotationWorkstationExtensionPlugin::startAnnotation(float x, float y, const std::string& type) {
  if (_generatedAnnotation) {
    return;
  }

  std::shared_ptr<Annotation> annot = std::make_shared<Annotation>();
  annot->addCoordinate(x / m_viewer_->getSceneScale(), y / m_viewer_->getSceneScale());
  if (type == "dotannotation") {
    annot->setType(Annotation::Type::DOT);
    _generatedAnnotation = new DotQtAnnotation(annot, this, m_viewer_->getSceneScale());
  }
  else if (type == "polyannotation") {
    annot->setType(Annotation::Type::POLYGON);
    PolyQtAnnotation* temp = new PolyQtAnnotation(annot, this, m_viewer_->getSceneScale());
    temp->setInterpolationType("linear");
    _generatedAnnotation = temp;
  }
  else if (type == "splineannotation") {
    annot->setType(Annotation::Type::SPLINE);
    PolyQtAnnotation* temp = new PolyQtAnnotation(annot, this, m_viewer_->getSceneScale());
    temp->setInterpolationType("spline");
    _generatedAnnotation = temp;    
  }
  else if (type == "pointsetannotation") {
    annot->setType(Annotation::Type::POINTSET);
    PointSetQtAnnotation* temp = new PointSetQtAnnotation(annot, this, m_viewer_->getSceneScale());
    _generatedAnnotation = temp;
  }
  else if (type == "measurementannotation") {
    annot->setType(Annotation::Type::MEASUREMENT);
    MeasurementQtAnnotation* temp = new MeasurementQtAnnotation(annot, this, m_viewer_->getSceneScale());
    _generatedAnnotation = temp;
  }
  else if (type == "rectangleannotation") {
    annot->setType(Annotation::Type::RECTANGLE);
    PolyQtAnnotation* temp = new PolyQtAnnotation(annot, this, m_viewer_->getSceneScale());
    temp->setInterpolationType("linear");
    _generatedAnnotation = temp;
  }
  else {
    return;
  }
  if (_generatedAnnotation) {
    _treeWidget->clearSelection();
	m_viewer_->scene()->addItem(_generatedAnnotation);
    _generatedAnnotation->setZValue(20.);
    updateGeneratingAnnotationLabel(_generatedAnnotation);
    connect(_generatedAnnotation, SIGNAL(annotationChanged(QtAnnotation*)), this, SLOT(updateGeneratingAnnotationLabel(QtAnnotation*)));
  }
}

void AnnotationWorkstationExtensionPlugin::updateGeneratingAnnotationLabel(QtAnnotation* annotation) {
  if (annotation) {
    _currentAnnotationLine->setVisible(true);
    _currentAnnotationLabel->setVisible(true);
    _currentAnnotationHeaderLabel->setVisible(true);
    unsigned int nrPoints = annotation->getAnnotation()->getNumberOfPoints();
    float area = annotation->getAnnotation()->getArea();
    QString areaUnit(" pixels.");
    if (_currentPixelArea != 1.) {
      areaUnit = QString(" um<sup>2</sup>");
    }
    _currentAnnotationLabel->setText(QString("Total number of control points: ") + QString::number(nrPoints) + QString("<br/>") + QString("Total area: ") + QString::number(area * _currentPixelArea, 'g', 4) + areaUnit);
  }
  else {
    _currentAnnotationLine->setVisible(false);
    _currentAnnotationLabel->setVisible(false);
    _currentAnnotationHeaderLabel->setVisible(false);
  }
}

void AnnotationWorkstationExtensionPlugin::removeViewer(void)
{
	m_viewer_ui_items_.erase((PathologyViewer*)sender());
}

void AnnotationWorkstationExtensionPlugin::finishAnnotation(bool cancel) {
	if (m_instance_)
	{
		if (_generatedAnnotation) {
			_generatedAnnotation->finish();
			updateGeneratingAnnotationLabel(NULL);
			disconnect(_generatedAnnotation, SIGNAL(annotationChanged(QtAnnotation*)), this, SLOT(updateGeneratingAnnotationLabel(QtAnnotation*)));
			if (!cancel) {
				_generatedAnnotation->getAnnotation()->setName("Annotation " + QString::number(m_instance_information_->annotation_index).toStdString());
				m_instance_information_->annotation_index += 1;
				m_annotation_items_->annotations.append(_generatedAnnotation);
				m_instance_information_->annotation_service->getList()->addAnnotation(_generatedAnnotation->getAnnotation());

				addAnnotationToGUI_(_generatedAnnotation);

				_activeAnnotation = _generatedAnnotation;
				updateAnnotationToolTip(_activeAnnotation);
				connect(_activeAnnotation, SIGNAL(annotationChanged(QtAnnotation*)), this, SLOT(updateAnnotationToolTip(QtAnnotation*)));
				_generatedAnnotation = NULL;
			}
			else {
				_controller->GetMasterViewer()->scene()->removeItem(_generatedAnnotation);
				_generatedAnnotation->deleteLater();
				_generatedAnnotation = NULL;
			}
		}
	}
}

void AnnotationWorkstationExtensionPlugin::deleteAnnotation(QtAnnotation* annotation) {
  if (annotation) {
    if (_treeWidget) {
      QTreeWidgetItemIterator it(_treeWidget);
      while (*it) {
        if (annotation == (*it)->data(1, Qt::UserRole).value<QtAnnotation*>()) {
          if (m_viewer_) {
			  m_viewer_->scene()->removeItem(annotation);
          }
          if (m_instance_information_) {
            std::vector<std::shared_ptr<Annotation> > annots = m_instance_information_->annotation_service->getList()->getAnnotations();
            int annotInd = std::find(annots.begin(), annots.end(), annotation->getAnnotation()) - annots.begin();
			m_instance_information_->annotation_service->getList()->removeAnnotation(annotInd);
          }
          annotation->deleteLater();
          _annotToItem.remove(annotation);
          m_annotation_items_->annotations.removeOne(annotation);
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

void AnnotationWorkstationExtensionPlugin::deleteAnnotationGroup(QtAnnotationGroup* group) {
  if (_treeWidget) {
    QTreeWidgetItemIterator it(_treeWidget);
    while (*it) {
      if (group == (*it)->data(1, Qt::UserRole).value<QtAnnotationGroup* >()) {
        if ((*it)->childCount() > 0) {
          for (int i = (*it)->childCount() - 1; i >= 0; --i) {
            QTreeWidgetItem* itm = (*it)->child(i);
            if (QtAnnotation* annot = itm->data(1, Qt::UserRole).value<QtAnnotation*>()) {
              deleteAnnotation(annot);
            }
            else {
              QtAnnotationGroup* grp = itm->data(1, Qt::UserRole).value<QtAnnotationGroup* >();
              if (grp) {              
                deleteAnnotationGroup(grp);
              }
            }
          }
        }
        if (m_instance_information_) {
          std::vector<std::shared_ptr<AnnotationGroup> > groups = m_instance_information_->annotation_service->getList()->getGroups();
          int groupInd = std::find(groups.begin(), groups.end(), group->getAnnotationGroup()) - groups.begin(); 
		  m_instance_information_->annotation_service->getList()->removeGroup(groupInd);
        }
        m_annotation_items_->groups.removeOne(group);
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
    if ((*it)->data(1, Qt::UserRole).value<QtAnnotation*>() == annotation) {
      (*it)->setSelected(true);
      break;
    }
    ++it;
  }
}

void AnnotationWorkstationExtensionPlugin::removeAnnotationFromSelection(QtAnnotation* annotation) {
  QTreeWidgetItemIterator it(_treeWidget);
  while (*it) {
    if ((*it)->data(1, Qt::UserRole).value<QtAnnotation*>() == annotation)  {
      (*it)->setSelected(true);
      break;
    }
    ++it;
  }
}

QSet<QtAnnotation*> AnnotationWorkstationExtensionPlugin::getSelectedAnnotations() {
  return _selectedAnnotations;
}

void AnnotationWorkstationExtensionPlugin::prepareForViewerChange_(void)
{
}

void AnnotationWorkstationExtensionPlugin::setupNewViewer_(void)
{
	m_viewer_ = _controller->GetMasterViewer();
	if (m_viewer_)
	{
		if (m_viewer_ui_items_.find(m_viewer_) == m_viewer_ui_items_.end())
		{
			connect(m_viewer_,
				&PathologyViewer::destructorCalled,
				this,
				&AnnotationWorkstationExtensionPlugin::removeViewer);

			m_viewer_ui_items_.insert({ m_viewer_, { QList<QtAnnotation*>(), QList<QtAnnotationGroup*>() } });
		}

		m_annotation_items_ = &m_viewer_ui_items_[m_viewer_];
	}
	else
	{
		m_annotation_items_ = nullptr;
	}
}

void AnnotationWorkstationExtensionPlugin::addAnnotationToGUI_(QtAnnotation* annotation)
{
	QTreeWidgetItem* newAnnotation = new QTreeWidgetItem(_treeWidget);
	newAnnotation->setText(1, QString::fromStdString(annotation->getAnnotation()->getName()));
	newAnnotation->setText(2, QString::fromStdString(annotation->getAnnotation()->getTypeAsString()));
	newAnnotation->setFlags(newAnnotation->flags() & ~Qt::ItemIsDropEnabled);
	newAnnotation->setFlags(newAnnotation->flags() | Qt::ItemIsEditable);
	newAnnotation->setData(1, Qt::UserRole, QVariant::fromValue<QtAnnotation*>(annotation));
	newAnnotation->setSelected(true);
	int cHeight = _treeWidget->visualItemRect(newAnnotation).height();
	QPixmap iconPM(cHeight, cHeight);
	iconPM.fill(QColor("yellow"));
	QIcon color(iconPM);
	newAnnotation->setIcon(0, color);
	newAnnotation->setData(0, Qt::UserRole, QColor("#F4FA58"));

	annotation->setZValue(20.);
	annotation->getAnnotation()->setColor("#F4FA58");
	_annotToItem[annotation] = newAnnotation;

	_controller->GetMasterViewer()->scene()->addItem(annotation);	
	_treeWidget->resizeColumnToContents(0);
	_treeWidget->resizeColumnToContents(1);
}

void AnnotationWorkstationExtensionPlugin::addAnnotationGroupToGUI_(QtAnnotationGroup* group)
{
	m_annotation_items_->groups.append(group);
	QTreeWidgetItem* newAnnotationGroup = new QTreeWidgetItem(_treeWidget);
	newAnnotationGroup->setText(1, QString::fromStdString(group->getAnnotationGroup()->getName()));
	newAnnotationGroup->setText(2, "Group");
	newAnnotationGroup->setData(1, Qt::UserRole, QVariant::fromValue<QtAnnotationGroup* >(group));
	newAnnotationGroup->setFlags(newAnnotationGroup->flags() | Qt::ItemIsEditable);
	int cHeight = _treeWidget->visualItemRect(newAnnotationGroup).height();
	QPixmap iconPM(cHeight, cHeight);
	iconPM.fill(QColor("#64FE2E"));
	QIcon color(iconPM);
	newAnnotationGroup->setIcon(0, color);
	newAnnotationGroup->setData(0, Qt::UserRole, QColor("#64FE2E"));
	group->getAnnotationGroup()->setColor("#64FE2E");
	_treeWidget->resizeColumnToContents(0);
	_treeWidget->resizeColumnToContents(1);
}

QtAnnotation* AnnotationWorkstationExtensionPlugin::InitializeAnnotation_(std::shared_ptr<Annotation> annotation)
{
	QtAnnotation* qt_annotation = nullptr;
	float scenescale			= _controller->GetMasterViewer()->getSceneScale();

	switch (annotation->getType())
	{
		case Annotation::Type::DOT:			qt_annotation = new DotQtAnnotation(annotation, this, scenescale); break;
		case Annotation::Type::POLYGON:		qt_annotation = new PolyQtAnnotation(annotation, this, scenescale);		((PolyQtAnnotation*)qt_annotation)->setInterpolationType("linear"); break;
		case Annotation::Type::SPLINE:		qt_annotation = new PolyQtAnnotation(annotation, this, scenescale);		((PolyQtAnnotation*)qt_annotation)->setInterpolationType("spline"); break;
		case Annotation::Type::POINTSET:	qt_annotation = new PointSetQtAnnotation(annotation, this, scenescale); break;
		case Annotation::Type::MEASUREMENT: qt_annotation = new MeasurementQtAnnotation(annotation, this, scenescale); break;
		case Annotation::Type::RECTANGLE:	qt_annotation = new PolyQtAnnotation(annotation, this, scenescale);		((PolyQtAnnotation*)qt_annotation)->setInterpolationType("linear"); break;
	}

	return qt_annotation;
}