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

namespace ASAP
{
	/// <summary>
	/// Describes the tile information for a Document.
	/// </summary>
	struct ASAPLIB_EXPORT TileInformation
	{
		uint64_t							tile_size;
		uint64_t							top_level;
		std::vector<float>					downsamples;
		std::vector<std::vector<uint64_t>>	dimensions;
	};

	class ASAPLIB_EXPORT Document
	{
		public:
			Document(const boost::filesystem::path& filepath, const std::string& factory = "default");

			MultiResolutionImage& image(void);

			boost::filesystem::path GetFilepath(void) const;
			const TileInformation& GetTileInformation(void) const;
			std::weak_ptr<MultiResolutionImage> GetImage(void);

		private:
			boost::filesystem::path					m_filepath_;
			std::shared_ptr<MultiResolutionImage>	m_image_;
			TileInformation							m_tile_information_;

			void InitializeImage_(const boost::filesystem::path& filepath, const std::string& factory);
			void InitializeTileInformation_(void);
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENT__