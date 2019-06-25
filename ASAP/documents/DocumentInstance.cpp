#include "DocumentInstance.h"

namespace ASAP
{
	DocumentInstance::DocumentInstance(Document& document) : document(&document)
	{
		SetupInstance_();
	}

	DocumentInstance::DocumentInstance(std::shared_ptr<Document> document) : document(document)
	{
		SetupInstance_();
	}

	DocumentInstance::DocumentInstance(std::weak_ptr<Document> document) : document(document)
	{
		SetupInstance_();
	}

	void DocumentInstance::SetupInstance_(void)
	{
		const TileInformation& tile_info(this->document->GetTileInformation());
		this->current_level = tile_info.top_level;
		this->minimap_coverage.resize(tile_info.top_level + 1);
	}
}