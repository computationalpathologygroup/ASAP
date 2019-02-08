#include "StringConversions.h"

namespace ASAP::Misc
{
	// ##### String to WideString ##### //

	std::wstring StringToWideString(const std::string& string)
	{
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
		return converter.from_bytes(string);
	}

	std::vector<std::wstring> StringsToWideStrings(const std::vector<std::string>& strings)
	{
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

		std::vector<std::wstring> new_strings;
		for (const std::string& string : strings)
		{
			new_strings.push_back(converter.from_bytes(string));
		}
		return new_strings;
	}

	// ##### WideString to String ##### //

	std::string WideStringToString(const std::wstring& string)
	{
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
		return converter.to_bytes(string);
	}

	std::vector<std::string> WideStringsToStrings(const std::vector<std::wstring> strings)
	{
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

		std::vector<std::string> new_strings;
		for (const std::wstring& string : strings)
		{
			new_strings.push_back(converter.to_bytes(string));
		}
		return new_strings;
	}
}