#include "DataTable.h"

#include <algorithm>
#include <stdexcept>

namespace ASAP::Data
{
	DataTable::DataTable(void)
	{
	}

	DataTable::DataTable(std::vector<std::string> columns)
	{
		ConvertColumnsToLower_(columns);

		for (size_t column = 0; column < columns.size(); ++column)
		{
			m_column_order_.insert({ columns[column], column });
		}

		for (auto it = m_column_order_.begin(); it != m_column_order_.end(); ++it)
		{
			m_visible_columns_.push_back(it);
		}
	}

	DataTable::DataTable(std::vector<std::string> columns, const std::vector<bool>& visibility)
	{
		ConvertColumnsToLower_(columns);

		if (columns.size() != visibility.size())
		{
			throw std::runtime_error("Vectors are required to be of the same size.");
		}

		for (size_t column = 0; column < columns.size(); ++column)
		{
			std::map<std::string, size_t>::iterator column_entry = m_column_order_.insert({ columns[column], column }).first;
			if (visibility[column])
			{
				m_visible_columns_.push_back(column_entry);
			}
			else
			{
				m_invisible_columns_.push_back(column_entry);
			}
		}
	}

	DataTable::DataTable(const DataTable& other)
		: m_column_order_(other.m_column_order_), m_data_(other.m_data_)
	{
		for (auto& it : other.m_visible_columns_)
		{
			m_visible_columns_.push_back(m_column_order_.find(it->first));
		}

		for (auto& it : other.m_invisible_columns_)
		{
			m_invisible_columns_.push_back(m_column_order_.find(it->first));
		}
	}

	DataTable::DataTable(DataTable&& other)
		: m_column_order_(std::move(other.m_column_order_)), m_data_(std::move(other.m_data_)), m_visible_columns_(std::move(other.m_visible_columns_)), m_invisible_columns_(std::move(other.m_invisible_columns_))
	{
	}

	DataTable& DataTable::operator=(const DataTable& other)
	{
		m_column_order_ = other.m_column_order_;
		m_data_ = other.m_data_;

		for (auto& it : other.m_visible_columns_)
		{
			m_visible_columns_.push_back(m_column_order_.find(it->first));
		}

		for (auto& it : other.m_invisible_columns_)
		{
			m_invisible_columns_.push_back(m_column_order_.find(it->first));
		}
		return *this;
	}

	DataTable& DataTable::operator=(DataTable&& other)
	{
		m_column_order_ = std::move(other.m_column_order_);
		m_data_ = std::move(other.m_data_);
		m_visible_columns_ = std::move(other.m_visible_columns_);
		m_invisible_columns_ = std::move(other.m_invisible_columns_);
		return *this;
	}

	void DataTable::Clear(void)
	{
		m_data_.clear();
	}

	bool DataTable::IsInitialized(void)
	{
		return m_column_order_.size() > 0 && (m_visible_columns_.size() > 0 || m_invisible_columns_.size() > 0);
	}

	std::vector<const std::string*> DataTable::At(const size_t index, const FIELD_SELECTION field_selection) const
	{
		std::vector<const std::string*> record;
		if (field_selection == FIELD_SELECTION::ALL)
		{
			record.resize((m_column_order_.size()));
			for (size_t column = 0; column < m_column_order_.size(); ++column)
			{
				record[column] = &m_data_[(m_column_order_.size() * index) + column];
			}
		}
		else if (field_selection == FIELD_SELECTION::VISIBLE)
		{
			record.resize((m_visible_columns_.size()));
			for (size_t column = 0; column < m_visible_columns_.size(); ++column)
			{
				record[column] = &m_data_[(m_column_order_.size() * index) + m_visible_columns_[column]->second];
			}
		}
		else
		{
			record.resize((m_invisible_columns_.size()));
			for (size_t column = 0; column < m_invisible_columns_.size(); ++column)
			{
				record[column] = &m_data_[(m_column_order_.size() * index) + m_invisible_columns_[column]->second];
			}
		}
		return record;
	}

	std::vector<const std::string*> DataTable::At(const size_t index, const std::vector<std::string> fields) const
	{
		std::vector<const std::string *> record(fields.size());
		for (size_t field = 0; field < fields.size(); ++field)
		{
			auto entry = m_column_order_.find(fields[field]);
			if (entry == m_column_order_.end())
			{
				throw std::runtime_error("Requested field isn't present within the table.");
			}
			record[field] = &m_data_[(m_column_order_.size() * index) + entry->second];
		}
		return record;
	}

	void DataTable::Insert(const std::vector<std::string>& record)
	{
		if (record.size() != m_column_order_.size())
		{
			throw std::runtime_error("Size mismatch.");
		}

		m_data_.insert(m_data_.end(), record.begin(), record.end());
	}

	size_t DataTable::GetRecordCount(void) const
	{
		return Size();
	}

	size_t DataTable::GetColumnCount(void) const
	{
		return m_column_order_.size();
	}

	size_t DataTable::GetColumnIndex(const std::string column) const
	{
		auto iterator = m_column_order_.find(column);
		if (iterator != m_column_order_.end())
		{
			return iterator->second;
		}
		else
		{
			throw std::runtime_error("Column " + column + " not found.");
		}
	}

	size_t DataTable::GetVisibleColumnCount(void) const
	{
		return m_visible_columns_.size();
	}

	size_t DataTable::GetInvisibleColumnCount(void) const
	{
		return m_invisible_columns_.size();
	}

	std::set<std::string> DataTable::GetColumnNames(const FIELD_SELECTION selection) const
	{
		std::set<std::string> header;
		if (selection == FIELD_SELECTION::ALL)
		{
			for (const auto& entry : m_column_order_)
			{
				header.insert(entry.first);
			}
		}
		else if (selection == FIELD_SELECTION::INVISIBLE)
		{
			for (const std::map<std::string, size_t>::iterator& it : m_invisible_columns_)
			{
				header.insert(it->first);
			}
		}
		else
		{
			for (size_t column = 0; column < m_visible_columns_.size(); ++column)
			{
				header.insert(m_visible_columns_[column]->first);
			}
		}
		return header;
	}

	void DataTable::SetColumnAsInvisible(const std::string column)
	{
		for (size_t col = 0; col < m_visible_columns_.size(); ++col)
		{
			if (m_visible_columns_[col]->first == column)
			{
				m_invisible_columns_.push_back(m_column_order_.find(column));
				m_visible_columns_.erase(m_visible_columns_.begin() + col);
			}
		}
	}

	void DataTable::SetColumnAsVisible(const std::string column)
	{
		for (size_t col = 0; col < m_invisible_columns_.size(); ++col)
		{
			if (m_invisible_columns_[col]->first == column)
			{
				m_visible_columns_.push_back(m_column_order_.find(column));
				m_invisible_columns_.erase(m_invisible_columns_.begin() + col);
			}
		}
	}

	size_t DataTable::Size(void) const
	{
		return (!m_data_.empty() && !m_column_order_.empty()) ? m_data_.size() / m_column_order_.size() : 0;
	}

	void DataTable::ConvertColumnsToLower_(std::vector<std::string>& columns)
	{
		for (std::string& column : columns)
		{
			std::transform(column.begin(), column.end(), column.begin(), ::tolower);
		}
	}
}