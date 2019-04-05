#include "DirectorySource.h"

#include "multiresolutionimageinterface/MultiResolutionImageFactory.h"

namespace ASAP::Data
{
	DirectorySource::DirectorySource(const std::string directory_path) : m_images_(GetImageFilelist_(directory_path))
	{
	}

	WorklistSourceInterface::SourceType DirectorySource::GetSourceType(void)
	{
		return WorklistSourceInterface::SourceType::FILELIST;
	}

	size_t DirectorySource::AddWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t DirectorySource::UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::vector<std::string> images, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t DirectorySource::DeleteWorklistRecord(const std::string& worklist_index, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t DirectorySource::GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectorySource::GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectorySource::GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectorySource::GetImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		receiver(m_images_, 0);
		return 0;
	}

	std::set<std::string> DirectorySource::GetWorklistHeaders(const DataTable::FIELD_SELECTION selectionL)
	{
		return std::set<std::string>();
	}

	std::set<std::string> DirectorySource::GetPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> DirectorySource::GetStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	std::set<std::string> DirectorySource::GetImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return std::set<std::string>();
	}

	size_t DirectorySource::GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		receiver(boost::filesystem::path(*m_images_.At(std::stoi(image_index), { "location" })[0]));
		return 0;
	}

	size_t DirectorySource::GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		receiver(boost::filesystem::path(*m_images_.At(std::stoi(image_index), { "location" })[0]));
		return 0;
	}

	DataTable DirectorySource::GetImageFilelist_(const std::string directory_path)
	{
		std::set<std::string> allowed_extensions = MultiResolutionImageFactory::getAllSupportedExtensions();	
		DataTable images({ "id", "location", "title" });

		boost::filesystem::path directory(directory_path);
		boost::filesystem::directory_iterator end_it;
		for (boost::filesystem::directory_iterator it(directory); it != end_it; ++it)
		{
			if (boost::filesystem::is_regular_file(it->path()) && (allowed_extensions.find(it->path().extension().string().substr(1)) != allowed_extensions.end()))
			{
				images.Insert({ std::to_string(images.Size()), it->path().string(), it->path().filename().string() });
			}
		}
		return images;
	}
}