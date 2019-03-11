#include "WorklistSourceInterface.h"

namespace ASAP::Data
{
	std::vector<std::string> WorklistSourceInterface::GetRequiredParameterFields(void)
	{
		// Empty vector signifies that no additional fields are required. This method should be overriden
		// if more information than just the location is required.
		return std::vector<std::string>();
	}

	void WorklistSourceInterface::CancelTask(const size_t id)
	{
		// Must be overridden incase the adapting class deals with asynchronous tasks. 
	}
}