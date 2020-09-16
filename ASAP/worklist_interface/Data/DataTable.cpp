#include "DataTable.h"

#include <algorithm>
#include <stdexcept>

namespace ASAP
{
	DataTable::DataTable(void)
	{
	}

	DataTable::DataTable(std::vector<std::string> columns)
	{
		convertColumnsToLower(columns);

		for (size_t column = 0; column < columns.size(); ++column)
		{
			m_column_order.insert({ columns[column], column });
		}

		for (auto it = m_column_order.begin(); it != m_column_order.end(); ++it)
		{
			m_visible_columns.push_back(it);
		}
	}

	DataTable::DataTable(std::vector<std::string> columns, const std::vector<bool>& visibility)
	{
		convertColumnsToLower(columns);

		if (columns.size() != visibility.size())
		{
			throw std::runtime_error("Vectors are required to be of the same size.");
		}

		for (size_t column = 0; column < columns.size(); ++column)
		{
			std::map<std::string, size_t>::iterator column_entry = m_column_order.insert({ columns[column], column }).first;
			if (visibility[column])
			{
				m_visible_columns.push_back(column_entry);
			}
			else
			{
				m_invisible_columns.push_back(column_entry);
			}
		}
	}

	DataTable::DataTable(const DataTable& other)
		: m_column_order(other.m_column_order), m_data(other.m_data)
	{
		for (auto& it : other.m_visible_columns)
		{
			m_visible_columns.push_back(m_column_order.find(it->first));
		}

		for (auto& it : other.m_invisible_columns)
		{
			m_invisible_columns.push_back(m_column_order.find(it->first));
		}
	}

	DataTable::DataTable(DataTable&& other) noexcept
		: m_column_order(std::move(other.m_column_order)), m_data(std::move(other.m_data)), m_visible_columns(std::move(other.m_visible_columns)), m_invisible_columns(std::move(other.m_invisible_columns)) 
	{
	}

	DataTable& DataTable::operator=(const DataTable& other)
	{
		m_column_order = other.m_column_order;
		m_data = other.m_data;

		for (auto& it : other.m_visible_columns)
		{
			m_visible_columns.push_back(m_column_order.find(it->first));
		}

		for (auto& it : other.m_invisible_columns)
		{
			m_invisible_columns.push_back(m_column_order.find(it->first));
		}
		return *this;
	}

	DataTable& DataTable::operator=(DataTable&& other)
	{
		m_column_order = std::move(other.m_column_order);
		m_data = std::move(other.m_data);
		m_visible_columns = std::move(other.m_visible_columns);
		m_invisible_columns = std::move(other.m_invisible_columns);
		return *this;
	}

	void DataTable::clear(void)
	{
		m_data.clear();
	}

	bool DataTable::isInitialized(void)
	{
		return m_column_order.size() > 0 && (m_visible_columns.size() > 0 || m_invisible_columns.size() > 0);
	}

	std::vector<const std::string*> DataTable::at(const size_t index, const FIELD_SELECTION field_selection) const
	{
		std::vector<const std::string*> record;
		if (field_selection == FIELD_SELECTION::ALL)
		{
			record.resize((m_column_order.size()));
			for (size_t column = 0; column < m_column_order.size(); ++column)
			{
				record[column] = &m_data[(m_column_order.size() * index) + column];
			}
		}
		else if (field_selection == FIELD_SELECTION::VISIBLE)
		{
			record.resize((m_visible_columns.size()));
			for (size_t column = 0; column < m_visible_columns.size(); ++column)
			{
				record[column] = &m_data[(m_column_order.size() * index) + m_visible_columns[column]->second];
			}
		}
		else
		{
			record.resize((m_invisible_columns.size()));
			for (size_t column = 0; column < m_invisible_columns.size(); ++column)
			{
				record[column] = &m_data[(m_column_order.size() * index) + m_invisible_columns[column]->second];
			}
		}
		return record;
	}

	std::vector<const std::string*> DataTable::at(const size_t index, const std::vector<std::string> fields) const
	{
		std::vector<const std::string *> record(fields.size());
		for (size_t field = 0; field < fields.size(); ++field)
		{
			auto entry = m_column_order.find(fields[field]);
			if (entry == m_column_order.end())
			{
				throw std::runtime_error("Requested field isn't present within the table.");
			}
			record[field] = &m_data[(m_column_order.size() * index) + entry->second];
		}
		return record;
	}

	void DataTable::insert(const std::vector<std::string>& record)
	{
		if (record.size() != m_column_order.size())
		{
			throw std::runtime_error("Size mismatch.");
		}

		m_data.insert(m_data_.end(), record.begin(), record.end());
	}

	size_t DataTable::getRecordCount(void) const
	{
		return size();
	}

	size_t DataTable::getColumnCount(void) const
	{
		return m_column_order.size();
	}

	size_t DataTable::getColumnIndex(const std::string column) const
	{
		auto iterator = m_column_order.find(column);
		if (iterator != m_column_order.end())
		{
			return iterator->second;
		}
		else
		{
			throw std::runtime_error("Column " + column + " not found.");
		}
	}

	size_t DataTable::getVisibleColumnCount(void) const
	{
		return m_visible_columns.size();
	}

	size_t DataTable::getInvisibleColumnCount(void) const
	{
		return m_invisible_columns.size();
	}

	std::set<std::string> DataTable::getColumnNames(const FIELD_SELECTION selection) const
	{
		std::set<std::string> header;
		if (selection == FIELD_SELECTION::ALL)
		{
			for (const auto& entry : m_column_order)
			{
				header.insert(entry.first);
			}
		}
		else if (selection == FIELD_SELECTION::INVISIBLE)
		{
			for (const std::map<std::string, size_t>::iterator& it : m_invisible_columns)
			{
				header.insert(it->first);
			}
		}
		else
		{
			for (size_t column = 0; column < m_visible_columns.size(); ++column)
			{
				header.insert(m_visible_columns[column]->first);
			}
		}
		return header;
	}

	void DataTable::setColumnAsInvisible(const std::string column)
	{
		for (size_t col = 0; col < m_visible_columns.size(); ++col)
		{
			if (m_visible_columns[col]->first == column)
			{
				m_invisible_columns.push_back(m_column_order.find(column));
				m_visible_columns.erase(m_visible_columns.begin() + col);
			}
		}
	}

	void DataTable::setColumnAsVisible(const std::string column)
	{
		for (size_t col = 0; col < m_invisible_columns.size(); ++col)
		{
			if (m_invisible_columns[col]->first == column)
			{
				m_visible_columns.push_back(m_column_order.find(column));
				m_invisible_columns.erase(m_invisible_columns.begin() + col);
			}
		}
	}

	size_t DataTable::size(void) const
	{
		return (!m_data.empty() && !m_column_order.empty()) ? m_data.size() / m_column_order.size() : 0;
	}

	void DataTable::convertColumnsToLower(std::vector<std::string>& columns)
	{
		for (std::string& column : columns)
		{
			std::transform(column.begin(), column.end(), column.begin(), ::tolower);
		}
	}
}