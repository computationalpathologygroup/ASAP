#include "XmlRepository.h"
#include "Annotation.h"
#include "AnnotationGroup.h"
#include "AnnotationList.h"
#include "core/Point.h"
#include <vector>
#include <string>
#include <stdio.h>
#include "pugixml.hpp"

XmlRepository::XmlRepository(AnnotationList* list) : 
  Repository(list)
{
}

bool XmlRepository::save() const
{
  if (!_list) {
    return false;
  }
	pugi::xml_document xml;
  pugi::xml_node root = xml.append_child("ASAP_Annotations");
  pugi::xml_node nodeAnnotations = root.append_child("Annotations");
  pugi::xml_node nodeGroups = root.append_child("AnnotationGroups");

  std::vector<Annotation*> annotations = _list->getAnnotations();
  for (std::vector<Annotation*>::const_iterator it = annotations.begin(); it != annotations.end(); ++it) 
	{
		saveAnnotation(*it, &nodeAnnotations);
	}

  std::vector<AnnotationGroup*> groups = _list->getGroups();
  for (std::vector<AnnotationGroup*>::const_iterator it = groups.begin(); it != groups.end(); ++it)
  {
    saveGroup(*it, &nodeGroups);
  }

  return xml.save_file(_source.c_str());
}

void XmlRepository::saveAnnotation(const Annotation* annotation, pugi::xml_node* node) const
{
	pugi::xml_node nodeAnnotation = node->append_child("Annotation");
  pugi::xml_attribute attributeName = nodeAnnotation.append_attribute("Name");
  attributeName.set_value(annotation->getName().c_str());
  pugi::xml_attribute attributeType = nodeAnnotation.append_attribute("Type");
  attributeType.set_value(annotation->getTypeAsString().c_str());
  pugi::xml_attribute attributeGroup = nodeAnnotation.append_attribute("PartOfGroup");
  if (annotation->getGroup()) {
    attributeGroup.set_value(annotation->getGroup()->getName().c_str());
  }
  else {
    attributeGroup.set_value("None");
  }
  pugi::xml_attribute attributeColor = nodeAnnotation.append_attribute("Color");
  attributeColor.set_value(annotation->getColor().c_str());

  pugi::xml_node nodeCoordinates = nodeAnnotation.append_child("Coordinates");
  std::vector<Point> coordinates = annotation->getCoordinates();
  for (std::vector<Point>::const_iterator it = coordinates.begin(); it != coordinates.end(); ++it) {
    pugi::xml_node nodeCoordinate = nodeCoordinates.append_child("Coordinate");
    pugi::xml_attribute attributeOrder = nodeCoordinate.append_attribute("Order");
    attributeOrder.set_value(static_cast<int>(it - coordinates.begin()));
    pugi::xml_attribute attributeX = nodeCoordinate.append_attribute("X");
    attributeX.set_value(it->getX());
    pugi::xml_attribute attributeY = nodeCoordinate.append_attribute("Y");
    attributeY.set_value(it->getY());
  }
}

void XmlRepository::saveGroup(const AnnotationGroup* group, pugi::xml_node* node) const
{
  pugi::xml_node nodeGroup = node->append_child("Group");
  pugi::xml_attribute attributeName = nodeGroup.append_attribute("Name");
  attributeName.set_value(group->getName().c_str());
  pugi::xml_attribute attributeGroup = nodeGroup.append_attribute("PartOfGroup");
  if (group->getGroup()) {
    attributeGroup.set_value(group->getGroup()->getName().c_str());
  }
  else {
    attributeGroup.set_value("None");
  }
  pugi::xml_attribute attributeColor = nodeGroup.append_attribute("Color");
  attributeColor.set_value(group->getColor().c_str());

  pugi::xml_node nodeAttributes = nodeGroup.append_child("Attributes");
  std::map<std::string, std::string> attributes = group->getAttributes();
  for (std::map<std::string, std::string>::const_iterator it = attributes.begin(); it != attributes.end(); ++it) {
    pugi::xml_node nodeAttribute = nodeAttributes.append_child("Attribute");
    pugi::xml_attribute attribute = nodeAttribute.append_attribute(it->first.c_str());
    attribute.set_value(it->second.c_str());
  }
}

bool XmlRepository::load()
{
  if (!_list) {
    return false;
  }

  _list->removeAllAnnotations();
  _list->removeAllGroups();

	pugi::xml_document xml_doc;
  pugi::xml_parse_result tree = xml_doc.load_file(_source.c_str());
  pugi::xml_node root = xml_doc.child("ASAP_Annotations");
  if (root.empty()) {
    root = xml_doc.root();
  }
  std::map<std::string, AnnotationGroup*> nameToGroup;
  std::map<std::string, std::string> groupToParent;
  pugi::xml_node groups = root.child("AnnotationGroups");
  for (pugi::xml_node_iterator it = groups.begin(); it != groups.end(); ++it)
  {
    AnnotationGroup* group = new AnnotationGroup();
    std::string groupName = it->attribute("Name").value();
    group->setName(groupName);
    std::string groupColor = it->attribute("Color").value();
    if (!groupColor.empty()) {
      group->setColor(groupColor);
    }
    std::string parentGroupName = it->attribute("PartOfGroup").value();
    if (parentGroupName != "None") {
      groupToParent[groupName] = parentGroupName;
    }
    nameToGroup[groupName] = group;
    pugi::xml_node groupAttributes = it->child("Attributes");
    for (pugi::xml_node_iterator itAttNode = groupAttributes.begin(); itAttNode != groupAttributes.end(); ++itAttNode) {
      for (pugi::xml_attribute_iterator itAttAtrributes = itAttNode->attributes_begin(); itAttAtrributes != itAttNode->attributes_end(); ++itAttAtrributes) {
        group->setAttribute<std::string>(itAttAtrributes->name(), itAttAtrributes->value());
      }
    }
    _list->addGroup(group);
  }

  // Now add the parent groups to each group
  std::vector<AnnotationGroup*> grps = _list->getGroups();
  for (std::vector<AnnotationGroup*>::iterator it = grps.begin(); it != grps.end(); ++it) {
    if (groupToParent.find((*it)->getName()) != groupToParent.end()) {
      (*it)->setGroup(nameToGroup[groupToParent[(*it)->getName()]]);
    }
  }

  pugi::xml_node annotations = root.child("Annotations");
	for (pugi::xml_node_iterator it = annotations.begin(); it != annotations.end(); ++it)
	{
		Annotation* annotation = new Annotation();

		annotation->setName(it->attribute("Name").value());
    annotation->setTypeFromString(it->attribute("Type").value());
    std::string annotColor = it->attribute("Color").value();
    if (!annotColor.empty()) {
      annotation->setColor(annotColor);
    }
    std::string annotationGroup = it->attribute("PartOfGroup").value();
    if (annotationGroup != "None") {
      if (nameToGroup.find(annotationGroup) != nameToGroup.end()) {
        annotation->setGroup(nameToGroup[annotationGroup]);
      }
      else {
        // XML inconsistent
        _list->removeAllAnnotations();
        _list->removeAllGroups();
        return false;
      }
    }
		
		pugi::xml_node coordinates = it->child("Coordinates");
    std::vector<std::vector<double> > coordsInOrder(coordinates.select_nodes("./Coordinate").size(), std::vector<double>(2, 0.0));
		for (pugi::xml_node_iterator cit = coordinates.begin(); cit != coordinates.end(); ++cit)
		{
			pugi::xml_attribute attributeX = cit->attribute("X");
			pugi::xml_attribute attributeY = cit->attribute("Y");
      std::vector<double> coord;
      coord.push_back(attributeX.as_double());
      coord.push_back(attributeY.as_double());
      coordsInOrder[cit->attribute("Order").as_int()] = coord;
		}
    for (std::vector<std::vector<double> >::const_iterator it = coordsInOrder.begin(); it != coordsInOrder.end(); ++it) {
      annotation->addCoordinate((*it)[0], (*it)[1]);
    }
		_list->addAnnotation(annotation);
	}
}