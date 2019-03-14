#include "Document.h"

#include <stdexcept>

#include <multiresolutionimageinterface/MultiResolutionImageReader.h>

namespace ASAP::Documents
{
	Document::Document(const std::string& filepath, const std::string& factory) : m_filepath_(filepath), m_image_(nullptr)
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

	const std::string& Document::GetFilepath(void)
	{
		return m_filepath_;
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
}