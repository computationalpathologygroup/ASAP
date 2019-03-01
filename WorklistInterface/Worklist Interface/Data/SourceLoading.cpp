#include "SourceLoading.h"

#include <codecvt>
#include <stdexcept>

#include <boost/filesystem.hpp>

#include "GrandChallengeDataAcquisition.h"
#include "DirectoryDataAcquisition.h"
#include "FilelistDataAcquisition.h"
#include "../Misc/StringConversions.h"

using namespace ASAP::Misc;
using namespace ASAP::Networking;

namespace ASAP::Data
{
	std::unique_ptr<WorklistDataAcquisitionInterface> LoadDataSource(const std::string source_path, const std::unordered_map<std::string, std::string> parameters, Misc::TemporaryDirectoryTracker& temp_dir)
	{
		try
		{
			boost::filesystem::path potential_system_path(source_path);
			std::unique_ptr<Data::WorklistDataAcquisitionInterface> pointer;

			if (source_path.empty())
			{
				pointer = nullptr;
			}
			else if (boost::filesystem::is_regular_file(potential_system_path) && CheckParameters(parameters, FilelistDataAcquisition::GetRequiredParameterFields()))
			{
				// Create File Acquisition
				pointer = std::unique_ptr<Data::WorklistDataAcquisitionInterface>(new Data::FilelistDataAcquisition(source_path));
			}
			else if (boost::filesystem::is_directory(potential_system_path) && CheckParameters(parameters, DirectoryDataAcquisition::GetRequiredParameterFields()))
			{
				pointer = std::unique_ptr<Data::DirectoryDataAcquisition>(new Data::DirectoryDataAcquisition(source_path));
			}
			else if (CheckParameters(parameters, GrandChallengeDataAcquisition::GetRequiredParameterFields()))
			{
				web::http::client::http_client_config config;
				config.set_validate_certificates(!static_cast<bool>(parameters.find("ignore_certificate")->second[0]));

				Data::GrandChallengeURLInfo uri_info = Data::GrandChallengeDataAcquisition::GetStandardURI(Misc::StringToWideString(source_path));
				Django_Connection::Credentials credentials(Django_Connection::CreateCredentials(StringToWideString(parameters.find("token")->second), L"api/v1/"));
				pointer = std::unique_ptr<Data::WorklistDataAcquisitionInterface>(new Data::GrandChallengeDataAcquisition(uri_info, temp_dir, credentials, config));
			}

			return pointer;
		}
		catch (const std::exception& e)
		{
			throw std::runtime_error("Unable to open source: " + source_path);
		}
	}

	namespace
	{
		bool CheckParameters(const std::unordered_map<std::string, std::string> additional_params, const std::vector<std::string> required_params)
		{
			for (const std::string& param : required_params)
			{
				auto it = additional_params.find(param);
				if (it == additional_params.end())
				{
					return false;
				}
			}
			return true;
		}
	}
}