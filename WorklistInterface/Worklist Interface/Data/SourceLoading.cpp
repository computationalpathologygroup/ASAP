#include "SourceLoading.h"

#include <codecvt>
#include <stdexcept>

#include <boost/filesystem.hpp>

#include "DjangoDataAcquisition.h"
#include "DirectoryDataAcquisition.h"
#include "FilelistDataAcquisition.h"

namespace ASAP::Data
{
	std::unique_ptr<WorklistDataAcquisitionInterface> LoadDataSource(const std::string source_path, const std::unordered_map<std::string, std::string> additional_params)
	{
		try
		{
			boost::filesystem::path potential_system_path(source_path);
			std::unique_ptr<Data::WorklistDataAcquisitionInterface> pointer;

			if (source_path.empty())
			{
				pointer = nullptr;
			}
			else if (boost::filesystem::is_regular_file(potential_system_path))
			{
				// Create File Acquisition
				pointer = std::unique_ptr<Data::WorklistDataAcquisitionInterface>(new Data::FilelistDataAcquisition(source_path));
			}
			else if (boost::filesystem::is_directory(potential_system_path))
			{
				pointer = std::unique_ptr<Data::DirectoryDataAcquisition>(new Data::DirectoryDataAcquisition(source_path));
			}
			else
			{
				std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
				Data::DjangoRestURI uri_info = Data::DjangoDataAcquisition::GetStandardURI();
				uri_info.base_url = converter.from_bytes(source_path);

				pointer = std::unique_ptr<Data::WorklistDataAcquisitionInterface>(new Data::DjangoDataAcquisition(uri_info));
			}

			return pointer;
		}
		catch (const std::exception& e)
		{
			throw std::runtime_error("Unable to open source: " + source_path);
		}
	}
}