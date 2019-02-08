#include "TemporaryDirectoryTracker.h"

#include <stdexcept>

#include <boost/range/iterator_range.hpp>

namespace ASAP::Worklist::Misc
{
	TemporaryDirectoryTracker::TemporaryDirectoryTracker(const boost::filesystem::path directory, const TemporaryDirectoryConfiguration configuration) : m_configuration_(configuration), m_continue_(true), m_directory_(directory)
	{
		if (boost::filesystem::exists(m_directory_) && boost::filesystem::is_regular_file(m_directory_))
		{
			throw std::runtime_error("Unable to initialize a file as temporary directory.");
		}

		m_update_thread_ = std::thread(&TemporaryDirectoryTracker::Update_, this);
	}

	TemporaryDirectoryTracker::~TemporaryDirectoryTracker(void)
	{
		m_continue_ = false;
		m_update_thread_.join();

		if (m_configuration_.clean_on_deconstruct)
		{
			boost::filesystem::remove_all(m_directory_);
		}
	}

	TemporaryDirectoryConfiguration TemporaryDirectoryTracker::GetStandardConfiguration(void) const
	{
		return { true, true, 0, 5000 };
	}

	boost::filesystem::path TemporaryDirectoryTracker::GetAbsolutePath(void) const
	{
		return boost::filesystem::absolute(m_directory_);
	}

	std::vector<boost::filesystem::path> TemporaryDirectoryTracker::GetFilepaths(void) const
	{
		std::vector<boost::filesystem::path> filepaths;
		for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(m_directory_), { }))
		{
			filepaths.push_back(entry.path());
		}
		return filepaths;
	}

	uint64_t TemporaryDirectoryTracker::GetDirectorySizeInMb(void) const
	{
		uint64_t size = 0;
		for (boost::filesystem::recursive_directory_iterator it(m_directory_); it != boost::filesystem::recursive_directory_iterator(); ++it)
		{
			if (!boost::filesystem::is_directory(*it))
			{
				size += boost::filesystem::file_size(*it) / 1e-6;
			}
		}

		return size;
	}

	void TemporaryDirectoryTracker::Update_(void)
	{
		while (m_continue_)
		{

		}
	}
}