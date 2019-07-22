#include "DocumentInstance.h"

#include <sstream>

namespace ASAP
{
	DocumentInstance::DocumentInstance(Document& document, const size_t document_id, const uint16_t instance_id)
		: document(&document), document_id(std::to_string(document_id)), name(GetInstanceName_(document, instance_id))
	{
		SetupInstance_();
	}

	DocumentInstance::DocumentInstance(std::shared_ptr<Document> document, const size_t document_id, const uint16_t instance_id)
		: document(document), document_id(std::to_string(document_id)), name(GetInstanceName_(*document, instance_id))
	{
		SetupInstance_();
	}

	DocumentInstance::DocumentInstance(std::weak_ptr<Document> document, const size_t document_id, const uint16_t instance_id)
		: document(document), document_id(std::to_string(document_id)), name(GetInstanceName_(*this->document, instance_id))
	{
		SetupInstance_();
	}
	
	const PluginInformation* DocumentInstance::GetPluginInformation(const std::string& plugin) const
	{
		auto it = m_plugin_information_.find(plugin);
		if (it != m_plugin_information_.end())
		{
			return it->second.information.get();
		}
		else
		{
			return nullptr;
		}
	}

	bool DocumentInstance::HasPluginInformation(const std::string& plugin) const
	{
		return m_plugin_information_.find(plugin) != m_plugin_information_.end();
	}

	void DocumentInstance::SetPluginInformation(const std::string& plugin, PluginInformation* information, const bool allow_override)
	{
		if (!HasPluginInformation(plugin) || allow_override)
		{
			m_plugin_information_.insert({plugin, PluginState(information)});
		}
	}
	
	std::string DocumentInstance::GetInstanceName_(Document& document, const uint16_t instance_id)
	{
		std::stringstream instance_name;
		instance_name << document.GetFilepath().filename().string() << " (" << std::to_string(instance_id) << ")";
		return instance_name.str();
	}

	void DocumentInstance::SetupInstance_(void)
	{
		const TileInformation& tile_info(this->document->GetTileInformation());
		this->current_level = tile_info.top_level;
		this->minimap_coverage.resize(tile_info.top_level + 1);
	}
}