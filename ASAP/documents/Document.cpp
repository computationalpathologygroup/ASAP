#include "Document.h"

#include <stdexcept>

#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"
#include "multiresolutionimageinterface/OpenSlideImage.h"

namespace ASAP
{
	Document::Document(const std::string& filepath, const std::string& factory) : m_filepath_(filepath), m_image_(nullptr)
	{
		InitializeImage_(filepath, factory);
		InitializeTileInformation_();
		
		m_state_ = CreateDocumentInstance(*this);
	}

	boost::filesystem::path Document::GetFilepath(void) const
	{
		return m_filepath_;
	}

	MultiResolutionImage& Document::AccessImage(void)
	{
		return *m_image_;
	}

	DocumentState& Document::AccessState(void)
	{
		return m_state_;
	}

	const TileInformation Document::GetTileInformation(void) const
	{
		return m_tile_information_;
	}

	std::weak_ptr<MultiResolutionImage> Document::GetImage(void)
	{
		return m_image_;
	}

	PluginInformation* Document::GetPluginInformation(const std::string& plugin)
	{
		return m_plugin_information_[plugin].get();
	}

	bool Document::HasPluginInformation(const std::string& plugin)
	{
		return m_plugin_information_.find(plugin) != m_plugin_information_.end();
	}

	void Document::SetPluginInformation(const std::string& plugin, PluginInformation* information, const bool allow_override)
	{
		if (allow_override || m_plugin_information_.find(plugin) == m_plugin_information_.end())
		{
			m_plugin_information_.insert({ plugin, std::unique_ptr<PluginInformation>(information) });
		}
	}

	void Document::InitializeImage_(const std::string& filepath, const std::string& factory)
	{
		// Ensures parameters have been set.
		if (filepath.empty())
		{
			throw std::invalid_argument("Filepath is empty.");
		}
		if (factory.empty())
		{
			throw std::invalid_argument("No factory selected");
		}

		// Attempts to open the file.
		MultiResolutionImageReader reader;
		m_image_.reset(reader.open(filepath, factory));

		// Checks if the image is valid.
		if (!m_image_)
		{
			throw std::runtime_error("Invalid file type");
		}
		if (!m_image_->valid())
		{
			throw std::runtime_error("Unsupported file type version");
		}

		// TODO: Ask why this is being done.
		std::vector<unsigned long long> dimensions = m_image_->getLevelDimensions(m_image_->getNumberOfLevels() - 1);
	}

	void Document::InitializeTileInformation_(void)
	{
		m_tile_information_.tile_size	= 512;
		m_tile_information_.top_level	= m_image_->getNumberOfLevels() - 1;
		for (uint64_t level = m_tile_information_.top_level; level >= 0; --level)
		{
			std::vector<unsigned long long> lastLevelDimensions = m_image_->getLevelDimensions(level);
			if (lastLevelDimensions[0] > m_tile_information_.tile_size && lastLevelDimensions[1] > m_tile_information_.tile_size)
			{
				m_tile_information_.top_level = level;
				break;
			}
		}

		for (unsigned int i = 0; i < m_image_->getNumberOfLevels(); ++i)
		{
			m_tile_information_.downsamples.push_back(m_image_->getLevelDownsample(i));
			m_tile_information_.dimensions.push_back(m_image_->getLevelDimensions(i));
		}
	}
}