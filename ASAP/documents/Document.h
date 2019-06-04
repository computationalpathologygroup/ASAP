#ifndef __ASAP_DOCUMENTS_DOCUMENT__
#define __ASAP_DOCUMENTS_DOCUMENT__

#include "asaplib_export.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <QPainterPath>
#include <QRect>
#include <QRectF>

class MultiResolutionImage;

class PluginInformation
{
	
};

/*typedef std::map<uint32_t, std::map<int32_t, std::map<int32_t, uchar>>> CoverageMap;

struct TileInformation
{
	uint64_t							tile_size;
	uint64_t							last_level;
	uint64_t							last_render_level;
	QRect								last_FOV;
	std::vector<float>					downsamples;
	std::vector<std::vector<uint64_t>>	dimensions;
	CoverageMap							coverage;
};*/

struct DocumentState
{
	uint64_t					current_level;
	QRect						current_fov;
	//QRect						current_fov_tile;
	std::vector<QPainterPath>	minimap_coverage;
	float						scene_scale;
};

struct TileInformation
{
	uint64_t							tile_size;
	uint64_t							top_level;
	std::vector<float>					downsamples;
	std::vector<std::vector<uint64_t>>	dimensions;
};


namespace ASAP
{
	class Document
	{
		public:
			Document(const std::string& filepath, const std::string& factory = "default");

			boost::filesystem::path GetFilepath(void) const;
			MultiResolutionImage& AccessImage(void);
			DocumentState& AccessState(void);

			const TileInformation GetTileInformation(void);
			std::weak_ptr<MultiResolutionImage> GetImage(void);

			PluginInformation* GetPluginInformation(const std::string& plugin);
			bool HasPluginInformation(const std::string& plugin);
			void SetPluginInformation(const std::string& plugin, PluginInformation* information, const bool allow_override = false);

		private:
			boost::filesystem::path					m_filepath_;
			std::shared_ptr<MultiResolutionImage>	m_image_;
			DocumentState							m_state_;
			TileInformation							m_tile_information_;
			std::unordered_map<std::string, std::unique_ptr<PluginInformation>> m_plugin_information_;

			void InitializeImage_(const std::string& filepath, const std::string& factory);
			void InitializeTileInformation_(void);
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENT__