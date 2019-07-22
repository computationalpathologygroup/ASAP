#include "PluginState.h"

namespace ASAP
{
	PluginState::PluginState(PluginInformation* information) : information(information)
	{
	}

	PluginState::PluginState(const PluginState& other) : information(new PluginInformation(*other.information))
	{

	}

	PluginState::PluginState(PluginState&& other) : information(std::move(other.information))
	{
	}

	PluginState::~PluginState(void)
	{
	}

	PluginState& PluginState::operator=(const PluginState& other)
	{
		information = std::unique_ptr<PluginInformation>(new PluginInformation(*other.information));
		return *this;
	}

	PluginState& PluginState::operator=(PluginState&& other)
	{
		information = std::move(other.information);
		return *this;
	}

}