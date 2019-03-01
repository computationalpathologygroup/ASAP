#ifndef __ASAP_DATA_SOURCELOADING__
#define __ASAP_DATA_SOURCELOADING__

#include <memory>
#include <string>
#include <unordered_map>

#include "WorklistDataAcquisitionInterface.h"
#include "../Misc/TemporaryDirectoryTracker.h"

namespace ASAP::Data
{
	std::unique_ptr<WorklistDataAcquisitionInterface> LoadDataSource(const std::string source_path, const std::unordered_map<std::string, std::string> additional_params, Misc::TemporaryDirectoryTracker& temp_dir);

	namespace
	{
		bool CheckParameters(const std::unordered_map<std::string, std::string> additional_params, const std::vector<std::string> required_params);
	}
}
#endif // __ASAP_DATA_SOURCELOADING__