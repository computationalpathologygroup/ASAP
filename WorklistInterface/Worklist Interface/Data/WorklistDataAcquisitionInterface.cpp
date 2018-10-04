#include "WorklistDataAcquisitionInterface.h"

namespace ASAP::Worklist::Data
{
	void WorklistDataAcquisitionInterface::CancelTask(const size_t id)
	{
		// Must be overridden incase the adapting class deals with asynchronous tasks. 
	}
}