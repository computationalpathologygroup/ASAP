#ifndef __ASAP_DATA_DATATABLE__
#define __ASAP_DATA_DATATABLE__

#include <map>
#include <string>
#include <set>
#include <vector>

namespace ASAP::Data
{
	class DataTable
	{
		public:
			enum FIELD_SELECTION { ALL, VISIBLE, INVISIBLE };

			DataTable(void);
			DataTable(std::vector<std::string> columns);
			DataTable(std::vector<std::string> columns, const std::vector<bool>& visibility);

			DataTable(const DataTable& other);
			DataTable(DataTable&& other);
			DataTable& operator=(const DataTable& other);
			DataTable& operator=(DataTable&& other);

			void Clear(void);
			bool IsInitialized(void);

			std::vector<const std::string*> At(const size_t index, const FIELD_SELECTION field_selection = ALL) const;
			std::vector<const std::string*> At(const size_t index, const std::vector<std::string> fields) const;
			void Insert(const std::vector<std::string>& record);
			size_t GetRecordCount(void) const;
			size_t GetColumnCount(void) const;
			size_t GetColumnIndex(const std::string column) const;
			size_t GetVisibleColumnCount(void) const;
			size_t GetInvisibleColumnCount(void) const;
			std::set<std::string> GetColumnNames(const FIELD_SELECTION selection = FIELD_SELECTION::ALL) const;
			void SetColumnAsInvisible(const std::string column);
			void SetColumnAsVisible(const std::string column);
			size_t Size(void) const;

		private:
			std::map<std::string, size_t>                         m_column_order_;
			std::vector<std::map<std::string, size_t>::iterator>  m_visible_columns_;
			std::vector<std::map<std::string, size_t>::iterator>  m_invisible_columns_;
			std::vector<std::string>                              m_data_;

			void ConvertColumnsToLower_(std::vector<std::string>& columns);
	};
}
#endif // __ASAP_DATA_DATATABLE__