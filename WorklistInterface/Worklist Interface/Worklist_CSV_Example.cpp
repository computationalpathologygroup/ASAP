#include "worklist_csv_example.h"

#include <iostream>

#include <exception>
#include <fstream>

Worklist_CSV_Example::Worklist_CSV_Example(const std::vector<std::string> paths)
{
    std::vector<std::vector<std::string>> headers;
    std::vector<std::vector<bool>> is_meta_data;

    Parse_(paths[0], headers, is_meta_data, this->m_worklist_items_);
    Parse_(paths[1], headers, is_meta_data, this->m_patient_items_);
    Parse_(paths[2], headers, is_meta_data, this->m_study_items_);
    Parse_(paths[3], headers, is_meta_data, this->m_image_items_);

    m_tables_.push_back(DataTable(headers[0], is_meta_data[0]));
    m_tables_.push_back(DataTable(headers[1], is_meta_data[1]));
    m_tables_.push_back(DataTable(headers[2], is_meta_data[2]));
    m_tables_.push_back(DataTable(headers[3], is_meta_data[3]));
}

DataTable Worklist_CSV_Example::GetWorklistRecords(void)
{
    m_tables_[0].Clear();
    for (const auto& record : m_worklist_items_)
    {
         m_tables_[0].Insert(record.second);
    }
    return m_tables_[0];
}

DataTable Worklist_CSV_Example::GetPatientRecords(const size_t worklist_index)
{
    std::vector<std::string> header(m_tables_[1].GetColumnNames());
    size_t index_column = 0;
    for (size_t column = 0; column < header.size(); ++column)
    {
        if (header[column] == "worklist")
        {
            index_column = column;
        }
    }

    m_tables_[1].Clear();
    // Determines if worklist exists.
    auto worklist_entry = m_worklist_items_.find(worklist_index);
    if (worklist_entry != m_worklist_items_.end())
    {
        for (const auto& record: m_patient_items_)
        {
            if (std::stoi(record.second[index_column]) == worklist_index)
            {
                m_tables_[1].Insert(record.second);
            }
        }
    }
    return m_tables_[1];
}

DataTable Worklist_CSV_Example::GetStudyRecords(const size_t patient_index)
{
	std::vector<std::string> header(m_tables_[2].GetColumnNames());
	size_t index_column = 0;
	for (size_t column = 0; column < header.size(); ++column)
	{
		if (header[column] == "patient")
		{
			index_column = column;
		}
	}

	m_tables_[2].Clear();
	// Determines if worklist exists.
	auto worklist_entry = m_worklist_items_.find(patient_index);
	if (worklist_entry != m_worklist_items_.end())
	{
		for (const auto& record : m_study_items_)
		{
			if (std::stoi(record.second[index_column]) == patient_index)
			{
				m_tables_[2].Insert(record.second);
			}
		}
	}

    return m_tables_[2];
}

DataTable Worklist_CSV_Example::GetImageRecords(const size_t study_index)
{
	std::vector<std::string> header(m_tables_[3].GetColumnNames());
	size_t index_column = 0;
	for (size_t column = 0; column < header.size(); ++column)
	{
		if (header[column] == "study")
		{
			index_column = column;
		}
	}

	m_tables_[3].Clear();
	// Determines if worklist exists.
	auto worklist_entry = m_worklist_items_.find(study_index);
	if (worklist_entry != m_worklist_items_.end())
	{
		for (const auto& record : m_image_items_)
		{
			if (std::stoi(record.second[index_column]) == study_index)
			{
				m_tables_[3].Insert(record.second);
			}
		}
	}

    return m_tables_[3];
}

void Worklist_CSV_Example::Parse_(const std::string& location,
           std::vector<std::vector<std::string>>& headers,
           std::vector<std::vector<bool>>& is_meta,
           std::unordered_map<size_t, std::vector<std::string>>& items)
{
    std::ifstream reader;
    reader.open(location);
    bool header_set = false;
    bool meta_set = false;
    if (reader.is_open())
    {
        do
        {
            std::string line;
            line.resize(3000);
            reader.getline(&line[0], 3000);
            if (!header_set)
            {
                headers.push_back(ParseLine_(line));
                header_set = true;
            }
            else if (!meta_set)
            {
                is_meta.push_back(ParseBools_(ParseLine_(line)));
                meta_set = true;
            }
            else
            {
                std::vector<std::string> parsed_line(ParseLine_(line));
               items.insert({ std::stoi(parsed_line[0]), parsed_line });
            }
        }
        while (!reader.eof());
    }
    else
    {
        throw std::runtime_error("Can't open: " + location);
    }

    reader.close();
}

std::vector<std::string> Worklist_CSV_Example::GetPatientHeaders(void)
{
    return m_tables_[1].GetVisibleColumnNames();
}

std::vector<std::string> Worklist_CSV_Example::GetStudyHeaders(void)
{
    return m_tables_[2].GetVisibleColumnNames();
}

std::vector<std::string> Worklist_CSV_Example::GetImageHeaders(void)
{
    return m_tables_[3].GetVisibleColumnNames();
}

std::vector<bool> Worklist_CSV_Example::ParseBools_(std::vector<std::string> values)
{
    std::vector<bool> bools;
    for (const std::string& s : values)
    {
        if (s[0] == '0')
        {
            bools.push_back(false);
        }
        else
        {
            bools.push_back(true);
        }
    }
    return bools;
}

std::vector<std::string> Worklist_CSV_Example::ParseLine_(const std::string& source)
{
    char delimiter = ',';

    // Prepares the parsing variables.
    size_t start_position = 0;
    size_t end_position = source.find(delimiter);
    size_t size = source.size();

    // Iterates through the string, acquiring the various values.
    std::vector<std::string> values;
    values.reserve(std::count(source.begin(), source.end(), delimiter));
    while (end_position < size && end_position != std::string::npos)
    {
        std::string value(source.substr(start_position, end_position - start_position));
        values.push_back(std::move(value));
        start_position = end_position + 1;
        end_position = source.find(delimiter, start_position);
    }

    // Adds the remaining characters to a value entry.
    std::string value(source.substr(start_position, source.find_first_of('\0', start_position) - start_position));
    if (values.size() != std::count(source.begin(), source.end(), delimiter) + 1)
    {
        values.push_back(std::move(value));
    }

    return values;
}
