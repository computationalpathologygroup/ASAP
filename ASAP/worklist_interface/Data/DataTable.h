#ifndef __ASAP_DATA_DATATABLE__
#define __ASAP_DATA_DATATABLE__

#include <map>
#include <string>
#include <set>
#include <vector>

namespace ASAP
{
	class DataTable
	{
		public:
			enum FIELD_SELECTION { ALL, VISIBLE, INVISIBLE };

			DataTable(void);
			DataTable(std::vector<std::string> columns);
			DataTable(std::vector<std::string> columns, const std::vector<bool>& visibility);

			DataTable(const DataTable& other);
			DataTable(DataTable&& other) noexcept;
			DataTable& operator=(const DataTable& other);
			DataTable& operator=(DataTable&& other);

			void clear(void);
			bool isInitialized(void);

			std::vector<const std::string*> at(const size_t index, const FIELD_SELECTION field_selection = ALL) const;
			std::vector<const std::string*> at(const size_t index, const std::vector<std::string> fields) const;
			void insert(const std::vector<std::string>& record);
			size_t getRecordCount(void) const;
			size_t getColumnCount(void) const;
			size_t getColumnIndex(const std::string column) const;
			size_t getVisibleColumnCount(void) const;
			size_t getInvisibleColumnCount(void) const;
			std::set<std::string> getColumnNames(const FIELD_SELECTION selection = FIELD_SELECTION::ALL) const;
			void setColumnAsInvisible(const std::string column);
			void setColumnAsVisible(const std::string column);
			size_t size(void) const;

		private:
			std::map<std::string, size_t>                         m_column_order;
			std::vector<std::map<std::string, size_t>::iterator>  m_visible_columns;
			std::vector<std::map<std::string, size_t>::iterator>  m_invisible_columns;
			std::vector<std::string>                              m_data;

			void convertColumnsToLower(std::vector<std::string>& columns);
	};
}
#endif // __ASAP_DATA_DATATABLE__