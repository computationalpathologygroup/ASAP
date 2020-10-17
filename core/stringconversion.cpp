#pragma hdrstop

#include "stringconversion.h"
#include <codecvt>

namespace core {

std::wstring stringToWideString(const std::string& string)
{
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.from_bytes(string);
}

std::vector<std::wstring> stringsToWideStrings(const std::vector<std::string>& strings)
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

std::string wideStringToString(const std::wstring& string)
{
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.to_bytes(string);
}

std::vector<std::string> wideStringsToStrings(const std::vector<std::wstring> strings)
{
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

  std::vector<std::string> new_strings;
  for (const std::wstring& string : strings)
  {
    new_strings.push_back(converter.to_bytes(string));
  }
  return new_strings;
}

void lower(std::string &s)
{
  for (unsigned int i = 0; i < s.size(); i++) if ((s[i] > 64) && (s[i] < 91)) s[i] = s[i] + 32;
}

void upper(std::string &s)
{
  for (unsigned int i = 0; i < s.size(); i++) if ((s[i] > 96) && (s[i] < 123)) s[i] = s[i] - 32;
}

void trim(std::string &s)
{
  if (s.empty()) return;
  unsigned int pos1 = 0, pos2 = (unsigned int) s.size() - 1;
  while ((pos1 < s.size()) && s[pos1] < 33) ++pos1;
  while ((pos2 > 0) && s[pos2] < 33) --pos2;
  ++pos2;
  if (pos2 > pos1) s = std::string(s.begin() + pos1, s.begin() + pos2);
}

bool replaceAll(std::string &s, const std::string &olds, const std::string &news)
{
  bool replaced = false;
  size_t oldpos = 0, pos;
  while (oldpos < s.length())
  {
    pos = s.find(olds, oldpos);
    if (pos != std::string::npos)
    {
      replaced = true;
      s = std::string(s.begin(), s.begin() + pos) +
          news +
          std::string(s.begin() + pos + olds.length(), s.end());
      oldpos = pos + news.length();
    }
    else oldpos = s.length();
  }
  return replaced;
}

void split(const std::string &s, std::vector<std::string> &vs, const std::string &split)
{
  vs.clear();
  size_t i1, i2;
  i1 = 0;
  do
  {
    i2 = s.find(split, i1);
    if (i2 == std::string::npos) i2 = s.size();
    //string p(s.begin()+i1,s.begin()+i2);
    vs.push_back(s.substr(i1, i2 - i1));
    i1 = i2 + split.size();
  }
  while (i2 != s.size());
}

//use HTML style escaping
std::string escapeCharacter(char c)
{
  return "&#" + tostring(int(c)) + ";";
}

void escape(std::string &str, char toEscape)
{
  //cannot escape string that contains the escaped sequence already because the mapping is not one-to-one anymore
  std::string unescaped = str;
  unescape(unescaped, toEscape);
  if (unescaped != str)
  {
    throw std::exception();
  }
  //
  size_t pos;
  std::string escaped = escapeCharacter(toEscape);
  while ((pos = str.find(toEscape)) != std::string::npos)
  {
    str = str.replace(pos, 1, escaped);
  }

}
void unescape(std::string &str, char toUnescape)
{

  size_t pos;
  std::string escaped = escapeCharacter(toUnescape);
  while ((pos = str.find(escaped)) != std::string::npos)
  {
    str = str.replace(pos, escaped.size(), std::string(1, toUnescape));
  }

}
}