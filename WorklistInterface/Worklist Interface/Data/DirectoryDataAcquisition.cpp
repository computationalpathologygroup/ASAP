#include "DirectoryDataAcquisition.h"

#include <codecvt>
#include <stdexcept>
#include <locale>
#include <system_error>
#include <cstdio>

#include "JSON_Parsing.h"

namespace ASAP::Worklist::Data
{
	DirectoryDataAcquisition::DirectoryDataAcquisition(const std::string directory_path)
	{
	}

	WorklistDataAcquisitionInterface::SourceType DirectoryDataAcquisition::GetSourceType(void)
	{
		return WorklistDataAcquisitionInterface::SourceType::FILELIST;
	}

	size_t DirectoryDataAcquisition::GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetPatientRecords(const size_t worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetStudyRecords(const size_t patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t DirectoryDataAcquisition::GetImageRecords(const size_t study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		return 0;
	//	return m_directory_images_;
	}

	std::vector<std::string> DirectoryDataAcquisition::GetPatientHeaders(void)
	{
		return std::vector<std::string>();
	}

	std::vector<std::string> DirectoryDataAcquisition::GetStudyHeaders(void)
	{
		return std::vector<std::string>();
	}

	std::vector<std::string> DirectoryDataAcquisition::GetImageHeaders(void)
	{
		return std::vector<std::string>();
	}

	std::vector<std::string> DirectoryDataAcquisition::GetImageFilelist_(const std::string directory_path)
	{

	}
}