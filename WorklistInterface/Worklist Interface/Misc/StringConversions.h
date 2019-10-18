#ifndef __ASAP_MISC_STRINGCONVERSIONS__
#define __ASAP_MISC_STRINGCONVERSIONS__

#include <codecvt>
#include <locale>
#include <string> 
#include <vector>

namespace ASAP::Misc
{
	/// <summary>
	/// Converts a string into a wstring.
	/// </summary>
	/// <param name="string">The string to convert.</param>
	/// <returns>The converted string as wstring.</returns>
	std::wstring StringToWideString(const std::string& string);
	/// <summary>
	/// Converts a vector of strings into a wstrings.
	/// </summary>
	/// <param name="string">The strings to convert</param>
	/// <returns>The converted strings as wstrings.</returns>
	std::vector<std::wstring> StringsToWideStrings(const std::vector<std::string>& strings);

	/// <summary>
	/// Converts a wstring into a string.
	/// </summary>
	/// <param name="string">The wstring to convert.</param>
	/// <returns>The converted wstring as string.</returns>
	std::string WideStringToString(const std::wstring& string);
	/// <summary>
	/// Converts a vector of wstring into a string.
	/// </summary>
	/// <param name="string">The wstrings to convert.</param>
	/// <returns>The converted wstrings as strings.</returns>
	std::vector<std::string> WideStringsToStrings(const std::vector<std::wstring> strings);
}
#endif // __ASAP_MISC_STRINGCONVERSIONS__