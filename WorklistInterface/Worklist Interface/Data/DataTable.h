#ifndef DATATABLE_H
#define DATATABLE_H

#include <map>
#include <string>
#include <vector>

class DataTable
{
    public:
        enum FIELD_SELECTION { ALL, VISIBLE, INVISIBLE };
		
		DataTable(void);
        DataTable(const std::vector<std::string>& columns);
        DataTable(const std::vector<std::string>& columns, const std::vector<bool>& visibility);

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
        size_t GetVisibleColumnCount(void) const;
        size_t GetInvisibleColumnCount(void) const;
        std::vector<std::string> GetColumnNames(void) const;
        std::vector<std::string> GetVisibleColumnNames(void) const;
        std::vector<std::string> GetInvisibleColumnNames(void) const;
		void SetColumnAsInvisible(const std::string column);
		void SetColumnAsVisible(const std::string column);
		size_t Size(void) const;

    private:
        std::map<std::string, size_t>                         m_column_order_;
        std::vector<std::map<std::string, size_t>::iterator>  m_visible_columns_;
        std::vector<std::map<std::string, size_t>::iterator>  m_invisible_columns_;
        std::vector<std::string>                              m_data_;
};

#endif // DATATABLE_H
