 #ifndef __ASAP_DATA_SOURCELOADING__
#define __ASAP_DATA_SOURCELOADING__

#include <memory>
#include <string>
#include <unordered_map>

#include "WorklistDataAcquisitionInterface.h"

namespace ASAP::Data
{
	std::unique_ptr<WorklistDataAcquisitionInterface> LoadDataSource(const std::string source_path, const std::unordered_map<std::string, std::string> additional_params);
}
#endif // __ASAP_DATA_SOURCELOADING__