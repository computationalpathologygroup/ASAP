#include "DataTable.h"

#include <stdexcept>

DataTable::DataTable(void)
{
}

DataTable::DataTable(const std::vector<std::string>& columns)
{
    for (size_t column = 0; column < columns.size(); ++column)
    {
        m_column_order_.insert({columns[column], column});
    }

	for (auto it = m_column_order_.begin(); it != m_column_order_.end(); ++it)
	{
		m_visible_columns_.push_back(it);
	}
}

DataTable::DataTable(const std::vector<std::string>& columns, const std::vector<bool>& visibility)
{
    if (columns.size() != visibility.size())
    {
        throw std::runtime_error("Vectors are required to be of the same size.");
    }

    for (size_t column = 0; column < columns.size(); ++column)
    {
        std::map<std::string, size_t>::iterator column_entry = m_column_order_.insert({columns[column], column}).first;
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
	m_data_			= other.m_data_;

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
	m_column_order_			= std::move(other.m_column_order_);
	m_data_					= std::move(other.m_data_);
	m_visible_columns_		= std::move(other.m_visible_columns_);
	m_invisible_columns_	= std::move(other.m_invisible_columns_);
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

size_t DataTable::GetVisibleColumnCount(void) const
{
    return m_visible_columns_.size();
}

size_t DataTable:: GetInvisibleColumnCount(void) const
{
    return m_invisible_columns_.size();
}

std::vector<std::string> DataTable::GetColumnNames(void) const
{
    std::vector<std::string> header(m_column_order_.size());
    for (const auto& entry : m_column_order_)
    {
        header[entry.second] = entry.first;
    }
    return header;
}

std::vector<std::string> DataTable::GetVisibleColumnNames(void) const
{
    std::vector<std::string> header;
    for (size_t column = 0; column < m_visible_columns_.size(); ++column)
    {
        header.push_back(m_visible_columns_[column]->first);
    }
    return header;
}

std::vector<std::string> DataTable::GetInvisibleColumnNames(void) const
{
    std::vector<std::string> header;
    for (const std::map<std::string, size_t>::iterator& it : m_invisible_columns_)
    {
        header.push_back(it->first);
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
    return m_data_.size() / m_column_order_.size();
}
