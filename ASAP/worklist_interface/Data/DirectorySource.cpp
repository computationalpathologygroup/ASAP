#include "DirectorySource.h"

#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"

namespace ASAP
{
	DirectorySource::DirectorySource(const std::string directory_path) : m_images(getImageFilelist(directory_path))
	{
	}

	WorklistSourceInterface::SourceType DirectorySource::getSourceType(void)
	{
		return WorklistSourceInterface::SourceType::DIRECTORY;
	}

	size_t DirectorySource::addWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t DirectorySource::updateWorklistRecord(const std::string& worklist_index, const std::string title, const std::set<std::string> images, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t DirectorySource::deleteWorklistRecord(const std::string& worklist_index, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t DirectorySource::getWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectorySource::getPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectorySource::getStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectorySource::getImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		receiver(m_images, 0);
		return 0;
	}

	std::set<std::string> DirectorySource::getWorklistHeaders(const DataTable::FIELD_SELECTION selectionL)
	{
		return std::set<std::string>();
	}

	std::set<std::string> DirectorySource::getPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> DirectorySource::getStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> DirectorySource::getImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	size_t DirectorySource::getImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		receiver(boost::filesystem::path(*m_images.at(std::stoi(image_index), { "location" })[0]));
		return 0;
	}

	size_t DirectorySource::getImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		receiver(boost::filesystem::path(*m_images.at(std::stoi(image_index), { "location" })[0]));
		return 0;
	}

	DataTable DirectorySource::getImageFilelist(const std::string directory_path)
	{
		std::set<std::string> allowed_extensions = MultiResolutionImageFactory::getAllSupportedExtensions();	
		DataTable images({ "id", "location", "title" });

		boost::filesystem::path directory(directory_path);
		boost::filesystem::directory_iterator end_it;
		for (boost::filesystem::directory_iterator it(directory); it != end_it; ++it)
		{
			if (boost::filesystem::is_regular_file(it->path()) && (allowed_extensions.find(it->path().extension().string().substr(1)) != allowed_extensions.end()))
			{
				images.insert({ std::to_string(images.size()), it->path().string(), it->path().filename().string() });
			}
		}
		return images;
	}
}