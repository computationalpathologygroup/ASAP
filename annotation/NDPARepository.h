#ifndef ANNOTATIONNDPASERVICE_H
#define ANNOTATIONNDPASERVICE_H

#include "config/pathology_config.h"
#include "Repository.h"

class EXPORT_PATHOLOGYANNOTATION NDPARepository : public Repository {
public:
  NDPARepository(const std::shared_ptr<AnnotationList>& list);

  void setNDPISourceFile(const std::string& ndpiSourcefile);
  std::string NDPISourceFile() const;

  virtual bool save() const;
  virtual bool load();

private:
  std::string _ndpiSourceFile;

};

#endif