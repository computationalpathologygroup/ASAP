#include "DocumentInstance.h"

namespace ASAP
{
	DocumentInstance::DocumentInstance(Document& document) : document(document)
	{
		const TileInformation& tile_info(this->document.GetTileInformation());

		this->current_level = tile_info.top_level;
		this->minimap_coverage.resize(tile_info.top_level + 1);
	}
}