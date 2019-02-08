#include "WorklistDataAcquisitionInterface.h"

namespace ASAP::Data
{
	std::vector<std::string> WorklistDataAcquisitionInterface::GetRequiredParameterFields(void)
	{
		// Empty vector signifies that no additional fields are required. This method should be overriden
		// if more information than just the location is required.
		return std::vector<std::string>();
	}

	void WorklistDataAcquisitionInterface::CancelTask(const size_t id)
	{
		// Must be overridden incase the adapting class deals with asynchronous tasks. 
	}
}