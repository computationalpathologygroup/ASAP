#ifndef __ASAP_MISC_TEMPORARYDIRECTORYTRACKER__
#define __ASAP_MISC_TEMPORARYDIRECTORYTRACKER__

#include <thread>
#include <vector>

#include <boost/filesystem.hpp>

namespace ASAP::Misc
{
	// TODO: Perhaps expand this class to handle a set of directoryies

	/// <summary>
	/// Holds variables that modify the behaviour of the temporary directory tracker.
	/// Setting any of the integer variables to 0 will effectively mean infinite space or files.
	///
	/// Warning: allow_overflow allows for a single file to be bigger than the directory,
	/// setting this to false will remove a file if it exceeds the maximum size.
	/// </summary>
	struct TemporaryDirectoryConfiguration
	{
		bool		allow_overflow;
		bool		clean_on_deconstruct;
		uint32_t	max_files;
		uint64_t	max_size_in_mb;
	};

	class TemporaryDirectoryTracker
	{
		public:
			TemporaryDirectoryTracker(const boost::filesystem::path directory, const TemporaryDirectoryConfiguration configuration = GetStandardConfiguration());
			~TemporaryDirectoryTracker(void);

			static TemporaryDirectoryConfiguration GetStandardConfiguration(void);

			boost::filesystem::path GetAbsolutePath(void) const;
			std::vector<boost::filesystem::path> GetFilepaths(void) const;
			uint64_t GetDirectorySizeInMb(void) const;

		private:
			TemporaryDirectoryConfiguration m_configuration_;
			bool							m_continue_;
			boost::filesystem::path			m_directory_;
			std::thread						m_update_thread_;

			void Update_(void);
	};
}
#endif // __ASAP_MISC_TEMPORARYDIRECTORYTRACKER__