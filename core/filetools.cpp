/***************************************************************-*-c++-*-

  @COPYRIGHT@

  $Id: filetools.cpp,v 1.35 2007/02/12 21:07:23 bram Exp $

*************************************************************************/

#pragma hdrstop

#include "filetools.h"

#ifdef WIN32
#include <windows.h>
#include <tchar.h>
#include <shellapi.h>
const std::string dirsep("\\");
#define WIN32_LEAN_AND_MEAN
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0500
// Note: we now need Windows 2000 or above
#include <windows.h>
#else
#include <unistd.h>
const std::string dirsep("/");
#define MAX_PATH PATH_MAX
#endif

#include <stdio.h>
#include <fcntl.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <regex>
#include <algorithm>

namespace fs = std::filesystem;

namespace core
{

    void replaceAll(std::string& s, const std::string& item, const std::string& replacement) {
        size_t pos = 0;
        while ((pos = s.find(item, pos)) != std::string::npos) {
            s.replace(pos, item.size(), replacement);
            pos += replacement.size();
        }
    }

    std::vector<std::string> split(const std::string& input, const std::string& regex) {
        // passing -1 as the submatch index parameter performs splitting
        std::regex re(regex);
        std::sregex_token_iterator
            first{ input.begin(), input.end(), re, -1 },
            last;
        return { first, last };
    }

  bool fileExists(const std::string &name)
  {
    try
    {
      fs::path p(name);
      if (!fs::exists(p)) return false;
      if (fs::is_directory(p)) return false;
      return true;
    }
    catch ( ... )
    {
      // exception, so this is not an existing file
      return false;
    }
  }

//---------------------------------------------------------------------------
  bool dirExists(const std::string &name)
  {
    try
    {
      fs::path p(name);
      bool ex = fs::exists(p);
      if (!ex) return false;
      ex = fs::is_directory(p);
      return ex;
    }
    catch ( ... )
    {
      // exception, so this is not an existing directory
      return false;
    }
  }

//---------------------------------------------------------------------------
  long int fileSize(const std::string &name)
  {
    if (!fileExists(name)) return -1;
	long int length = static_cast<long int>(fs::file_size(name));
    return length;
  }
//---------------------------------------------------------------------------
  bool deleteFile(const std::string &name)
  {
    if (!fileExists(name)) return false;
    try
    {
      remove(fs::path(uniformSlashes(name)));
    }
    catch (...)
    {
      return false;
    }
    return true;
  }

  std::string uniformSlashes(const std::string &path)
  {
    std::string localpath = path;
#ifdef WIN32
    std::replace(localpath.begin(), localpath.end(), '/', '\\');
#else
    std::replace(localpath.begin(), localpath.end(), '\\', '/');
#endif
    return localpath;
  }

  bool deleteDir(const std::string &name, bool deleteNonEmpty)
  {
    if (!dirExists(name)) return false;
    if (!deleteNonEmpty)
      if (!emptyDir(name)) return false;
    return 0 != fs::remove_all(fs::path(uniformSlashes(name)));
  }


//---------------------------------------------------------------------------
  bool emptyDir(const std::string &name)
  {
    if (!dirExists(name)) return false;
    return fs::is_empty(fs::path(name));
  }

//---------------------------------------------------------------------------
  bool copyFile(const std::string &source, const std::string &target, bool overwrite, bool copyAttributes)
  {
    //copy_file throws: if !exists(source_file) || is_directory(source_file)
    //|| exists(target_file) || target_file.empty() || !exists(to_file_path/"..")).
    //we check these first and return false if needed

    //Check source
    if (!fileExists(source)) return false;

    //Check taget
    //Check taget
    if (!dirExists(extractFilePath(target)))
    {
      if (!createDirectory(extractFilePath(target)))
      {
        return false;
      }
    }
    else if (fileExists(target))
    {
      if (overwrite)
      {
        if (!deleteFile(target))
        {
          return false;
        }
      }
      else
      {
        return false;
      }
    }

    if (!copyAttributes)
    {
      //now we can safely copy
      //create_file( path(target));
      //this trick is needed to prevent attributes from being copied
      std::ifstream ifs(source.c_str());
      std::ofstream ofs(target.c_str());
      ofs << ifs.rdbuf();
      ofs.close();
      ifs.close();
    }
    else
    {
        fs::copy_file(fs::path(source), fs::path(target));
    }
    return fileExists(target);
  }


//---------------------------------------------------------------------------
  bool copyDirectory(
    const std::string &source,
    const std::string &target,
    const std::string &name,
    bool recurse,
    bool overwrite,
    bool copyAttributes)
  {
    //Check source
    if (!dirExists(source))
      return false;

    //Check target
    if (!dirExists(target))
    {
      if (!createDirectory(target))
      {
        return false;
      }
    }
    else if (!overwrite)
    {
      if (!emptyDir(target))
        return false;
    }


    //Get source files
    std::vector<std::string> files;
    getFiles(source, name, files, recurse);

    if (files.empty())
      return true;

    //Copy files
    std::string sourceFile, targetFile;
    const size_t sourceSize = source.size();

    for (std::vector<std::string>::iterator iter = files.begin(); iter != files.end(); ++iter)
    {
      sourceFile = (*iter);
      targetFile = target + sourceFile.substr(sourceSize, (*iter).size());
      if (!copyFile(sourceFile, targetFile, overwrite, copyAttributes))
        return false;
    }

    return true;

  }



//---------------------------------------------------------------------------
  bool renameFile(const std::string &source, const std::string &target)
  {
    //checks are inserted here, because rename may throw...
    std::string s(source), t(target);
    cleanFileName(s);
    cleanFileName(t);
    if (!fileExists(s)) return false;
    if (fileExists(t)) return false;
    std::string dir = extractFilePath(t);
    createDirectory(dir);
    //BvG: if target is on a different drive, source will also be deleted
    rename(fs::path(s), fs::path(t));
    return fileExists(t);
  }

//---------------------------------------------------------------------------
  std::string upOneLevel(const std::string &name)
  {
    if (isRoot(name))
      return name;

    std::string s(name);
    if (isOnlyDirectory(name))
      s = stripTrailingSlash(s);

    fs::path p(s);
    return p.parent_path().string();
  }

  std::string upMultipleLevels(const std::string &name, unsigned int nrOfLevels)
  {
    if (isRoot(name))
      return name;

    std::string s(name);
    if (isOnlyDirectory(name))
      s = stripTrailingSlash(s);

    fs::path p(s);
    if (nrOfLevels == 1)
      return p.parent_path().string();
    else
      return upMultipleLevels(p.parent_path().string(), nrOfLevels - 1);
  }

//---------------------------------------------------------------------------
  std::string extractFilePath(const std::string &name)
  {
    std::string s(name);
    if (isOnlyDirectory(s))
    {
      fs::path p(s);
      if (isRoot(name))
        return p.string();
      else
        return stripTrailingSlash(std::string(p.string()));
    }
    else
    {
      cleanFileName(s);
      fs::path p(s);
      if (isRoot(name)) // CM: needed for "\\\\machine\\" see testFileTools.cpp
        return p.string();
      return std::string(p.parent_path().string());
    }
  }

//---------------------------------------------------------------------------
  std::string extractFileName(const std::string &name)
  {
    std::string s(name);
    if (isOnlyDirectory(s)) return std::string();
    cleanFileName(s);
    fs::path p(s);
    if (isRoot(name)) return std::string();
    return p.filename().string();
  }

//---------------------------------------------------------------------------
  std::string extractLowestDirName(const std::string &name)
  {
    // Note that this function now ASSUMES!! a directory is given as input
    std::string s(name);
    // CM: if both lines below are use, it makes no sense, cleanDir adds "\\" and thus isOnlyDirectory will always succeed
//  cleanDirName(s);
//  if (! isOnlyDirectory(s)) return string();
    s = stripTrailingSlash(s);
    fs::path p(s);
    return p.filename().string();
  }


//---------------------------------------------------------------------------
  std::string extractBaseName(const std::string &name)
  {
    std::string s(name);
    if (isOnlyDirectory(name)) return std::string();
    cleanFileName(s);
    std::string res = extractFileName(s);
    size_t pos = res.rfind(".");
    if (pos != std::string::npos) res.resize(pos);
    return res;
  }


//---------------------------------------------------------------------------
  std::string extractFileExtension(const std::string &name)
  {
    std::string s = extractFileName(name);
    size_t pos = s.rfind('.');
    if (pos != std::string::npos) return std::string(s.begin() + pos + 1, s.end());
    return std::string();
  }

//---------------------------------------------------------------------------
  bool createDirectory(const std::string &dir)
  {
    std::string s(dir);
    cleanDirName(s);
    try
    {
      fs::path p(s);
      create_directories(p);
    }
    catch ( ... )
    {
    }
    return dirExists(s);
  }

//---------------------------------------------------------------------------
  std::string getPathRelativeToLocation(const std::string &_pathToAlter, const std::string &_fixedPath)
  {
//Make copies so we can change them for processing
    std::string pathToAlter = _pathToAlter;
    std::string fixedPath = _fixedPath;

//if a relative path cannot be written
    if ( (rootName(pathToAlter)).compare(rootName(fixedPath)) != 0 )
      return pathToAlter;
//if the two paths are the same
    if ( pathToAlter.compare(fixedPath) == 0)
      return std::string(".");

//if the path ends with a "\" then cut it off
    if (pathToAlter.find_last_of(dirsep) == pathToAlter.length() - 1)
      pathToAlter = pathToAlter.substr(0, pathToAlter.length() - 1);
    if (fixedPath.find_last_of(dirsep) == fixedPath.length() - 1)
      fixedPath = fixedPath.substr(0, fixedPath.length() - 1);


    std::string relativePath;
//If the entire fixed path is contained within the path to be given
    if ( pathToAlter.length() >= fixedPath.length()
         && pathToAlter.substr(0, fixedPath.length()).compare(fixedPath) == 0 )
    {
      //Tben we just need "./" and to append the remainder (non-matched part) of pathToAlter
      relativePath = std::string(".") + dirsep + pathToAlter.substr(fixedPath.length() + 1, pathToAlter.length() - fixedPath.length());
    }
//Otherwise we need at least one "../" step in the relative path
    else
    {
      relativePath = std::string("..") + dirsep;
      //step up a level in the fixed path and see if it is now contained within the pathToAlter
      std::string choppedFixed = upOneLevel(fixedPath);
      //As long as the "choppedFixed" is not contained in pathToAlter
      while ( pathToAlter.substr(0, choppedFixed.length()).compare(choppedFixed) != 0)
      {
        //Keep stepping up one level and appending "../" to relative path
        relativePath = relativePath + std::string("..") + dirsep;
        choppedFixed = upOneLevel(choppedFixed);
      }

      //if choppedFixed ends with a "\" then cut it off (e.g. if it is C:\ then '\' always appended)
      if (choppedFixed.find_last_of(dirsep) == choppedFixed.length() - 1)
        choppedFixed = choppedFixed.substr(0, choppedFixed.length() - 1);

      //A string to hold the path that we need to append to the "../../../" we made
      std::string endOfRelPath = std::string("");

      //Take the end part of "pathToAlter" (The part which doesn't match with choppedFixed)
      if (pathToAlter.length() > choppedFixed.length())
        endOfRelPath = pathToAlter.substr(choppedFixed.length() + 1, pathToAlter.length() - choppedFixed.length()) ;

      //Append this to the "../../../" part
      relativePath = relativePath + endOfRelPath;
    }
    return relativePath;
  }

//---------------------------------------------------------------------------
  std::string rootName(const std::string &spath)
  {
    fs::path p(spath);
    return p.root_name().string();
  }

  bool isRoot(const std::string &spath)
  {
    if (spath.empty()) return false;
    fs::path p(spath);
    std::string s1 = p.string();
    std::string s2 = p.root_path().string();
    return (s1 == s2);
  }

  bool isUNCPath(const std::string &spath)
  {
    std::string s = rootName(spath);
    if (s.size() < 3) return false;
    // Attention: Root is returned by boost as //ComputerName
    if ((s[0] == '/') && (s[1] == '/')) return true;
    return false;
  }

  bool isOnlyDirectory(const std::string &spath)
  {
    //a string can only signify a directory and not a file name if it ends
    // in \\ or / or : or if it is of the form \\\\machine or //machine
    if (spath.empty()) return false;
    char f = spath[spath.size()-1];
    char b = tolower(spath[0]);
    if (f == '/') return true;
    if (f == '\\') return true;
    if ((f == ':') && (spath.size() == 2) && ((b >= 'a') && (b <= 'z'))) return true;

    std::string s(spath);
    cleanDirName(s);
    try
    {
      fs::path p(s);
      if (p.filename() == p.root_name())
        return true;
    }
    catch ( ... )
    {
      // exception, so this is not an DIR
    }
    return false;
  }

//---------------------------------------------------------------------------
  bool isOnlyDirectoryTmpKeelin(const std::string &spath)
  {
    try
    {
      fs::path mypath(spath);
      if (fs::is_directory(mypath))
        return true;
    }
    catch ( ... )
    {
      // exception, so this is not an DIR
    }
    return false;
  }

//---------------------------------------------------------------------------
  std::string currentDirPath()
  {
    fs::path p = fs::current_path();
    return p.string();
  }

//---------------------------------------------------------------------------

  std::string stripTrailingSlash(std::string strPath)
  {
#ifdef WIN32
    char lastchar = strPath[strPath.size()-1];
    if (lastchar == '\\' ||  lastchar == '/')
    {
      strPath = strPath.substr(0, strPath.size() - 1);
    }
    else
    {
      std::string x = strPath.substr(strPath.size() - 2, 2);
      if (x == "\\." ||  lastchar == '/.')
      {
        strPath = strPath.substr(0, strPath.size() - 2);
      }
    }
#else
    char lastchar = strPath[strPath.size()-1];
    if (lastchar == '/')
    {
      strPath = strPath.substr(0, strPath.size() - 1);
    }
    else
    {
      std::string x = strPath.substr(strPath.size() - 2, 2);
      if (x == "/.")
      {
        strPath = strPath.substr(0, strPath.size() - 2);
      }
    }
#endif
    return strPath;
  }

  std::string completePath(const std::string &spath, const std::string &base)
  {
    if (spath.empty())
      return base;

    std::string cpath(spath);
    cleanDirName(cpath);
    fs::path pp(cpath);
    if (pp.is_absolute())
    {
      // CM: this fiddling about is based on the tests in testFileTools.cpp, maybe we should revise these tests?
      std::string h = pp.root_path().string();
      if (h.find("//") == 0)
      {
        if (spath[spath.length()-1] == dirsep[0])
          return pp.string();
        else
          return stripTrailingSlash(pp.string());
      }
      return stripTrailingSlash(pp.string());
    }

    std::string s(base);
    cleanDirName(s);
    fs::path pb(s);

    fs::path cp;
    if (! pp.empty())
    {
      if ((spath == "\\" || spath == "/") && pb.has_root_path())
        return pb.root_path().string();
      else
        cp = pb / pp;
    }
    else cp = pb;

    return stripTrailingSlash(cp.string());
  }

//---------------------------------------------------------------------------
//MN: WARNING, THIS FUNCTION HAS BEEN OBSERVED TO GENERATE FILESYSTEM EXCEPTIONS
//    IN THE WILD.
  void getFiles(
    const std::string &thepath,
    const std::string &name,
      std::vector<std::string> &v,
    bool recurse)
  {
    v.clear();
    std::string pa(thepath);
    if (pa.empty()) pa = ".";
    cleanDirName(pa);
#ifdef WIN32
    replaceAll(pa, "/", "\\");
    if (pa.empty()) pa = ".";
    if (pa[pa.size()-1] != '\\') pa += "\\";
#else
    replaceAll(pa, "\\", "/");
    if (pa.empty()) pa = ".";
    if (pa[pa.size()-1] != '/') pa += "/";
#endif
    if (!dirExists(pa)) return;

    std::vector<fs::path> vp;
    vp.push_back(fs::path(pa));
    unsigned int i = 0;

    //to create a proper regular expression, any non literal should be preceded by \:
    //All characters are literals except: ".", "|", "*", "?", "+", "(", ")", "{", "}",
    //"[", "]", "^", "$" and "\".
    //some of these characters are not allowed in filenames, ...
    //and * should be replaced by .* and ? should be replaced by .?
    std::string filename = name;

    replaceAll(filename, ".", "\\.");
    replaceAll(filename, "|", "\\|");
    replaceAll(filename, "|", "\\|");
    replaceAll(filename, "+", "\\+");
    replaceAll(filename, "(", "\\(");
    replaceAll(filename, ")", "\\)");
    replaceAll(filename, "[", "\\[");
    replaceAll(filename, "]", "\\]");
    replaceAll(filename, "{", "\\{");
    replaceAll(filename, "}", "\\}");
    replaceAll(filename, "$", "\\$");
    replaceAll(filename, "^", "\\^");
    replaceAll(filename, "*", ".*");
    replaceAll(filename, "?", ".?");
    std::regex e(filename, std::regex::icase);

    while (i < vp.size())
    {
      //now we collect all files in the directory and check if they match the pattern
      fs::directory_iterator end_itr; // default construction yields past-the-end
      std::string p = vp[i].string();
#ifdef WIN32
      if (p[p.size()-1] != '\\') p += "\\";
#else
      if (p[p.size()-1] != '/') p += "/";
#endif
      for (fs::directory_iterator itr(vp[i]); itr != end_itr; ++itr)
      {
        //BvG: the call to is_directory could raise an exception if
        //the directory is not accessible
        bool b;
        try
        {
          b = fs::is_directory(*itr);
        }
        catch ( ... )
        {
          // handler for any C++ exception
          b = true; // so we skip this one
          //libReport(eError,"Error in getFiles: is_directory %s\n",itr->string().c_str());
        }
        if ( !b )
        {
          std::string f = itr->path().filename().string();
          if (std::regex_match(f, e)) v.push_back(p + f);
        }
        else if (recurse)
        {

          std::string subdir = itr->path().string() + dirsep;
          vp.push_back(fs::path(subdir));
        }
      }
      ++i; // move to the next directory to process
    }

    //BvG: it seems some systems do not return files in alphabetical order (samba?)
    //so we sort the file names here
    sort(v.begin(), v.end());
  }

//---------------------------------------------------------------------------
  void getSubdirectories(
    const std::string &thepath,
      std::vector<std::string> &v,
    bool recurse)
  {
    v.clear();
    std::string pa = thepath;
    std::vector<fs::path> vp;
    fs::path p(pa);
    vp.push_back(p);
    v.push_back(p.string());

    unsigned int i = 0;
    while (i < vp.size())
    {
      //now we collect all subdirectories of path vp[i]
      fs::directory_iterator end_itr; // default construction yields past-the-end
      for (fs::directory_iterator itr(vp[i]); itr != end_itr; ++itr)
      {
        //BvG: the call to is_directory could raise an exception if
        //the directory is not accessible
        bool b;
        try
        {
          b = fs::is_directory(*itr);
        }
        catch ( ... )
        {
          // handler for any C++ exception
          b = false; // so we skip this one
          //libReport(eError,"Error in getSubdirectories: is_directory %s\n",itr->string().c_str());
        }
        if (b)
        {
          std::string subdir = itr->path().string();
          v.push_back(subdir);
          if (recurse)
            vp.push_back(fs::path(subdir));
        }
      }
      ++i; // move to the next directory to process
    } // while (i<vp.size())
  }

#ifdef WIN32
  void getSubdirectoriesWindows(
    const std::string &thepath,
    std::vector<std::string> &v,
    bool recurse)
  {
    WIN32_FIND_DATAA FindFileData;
    HANDLE hFind;

    hFind = FindFirstFileExA((thepath + "\\*.*").c_str(), FindExInfoStandard, &FindFileData, FindExSearchNameMatch, 0, 0);
    BOOL ok = 1;
    while (ok)
    {
      if (((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY) && (FindFileData.cFileName[0] != '.'))
      {
        v.push_back(thepath + "\\" + FindFileData.cFileName);
        if (recurse)
        {
          getSubdirectoriesWindows(thepath + "\\" + FindFileData.cFileName, v, recurse);
        }
      }
      ok = FindNextFileA(hFind, &FindFileData);
    }
    FindClose(hFind);
  }
#endif

//---------------------------------------------------------------------------
  void changeExtension(std::string &name, const std::string &newextension)
  {
    if (extractFileExtension(name) != "")
    {
      std::string path = extractFilePath(name);
      if ((!path.empty()) && (!isRoot(path)))
        path += dirsep;

      if (extractBaseName(name) != "")
      {
        name = path + extractBaseName(name);
        if (newextension != "") name +=  ".";
      }
      else name = path;
      name += newextension;
    }
    else
    {
      // this is for uniformity - it returns name in boost format
      // if name is "./dir" it returns ".\\dir" instead of "./dir"
      name = fs::path(name).string();
      if (!name.empty() && !isRoot(name))
        name = stripTrailingSlash(name);
    }
  }


//---------------------------------------------------------------------------
  void changeBaseName(std::string &name, const std::string &newbasename)
  {
    if (extractBaseName(name) != "")
    {
      std::string path =  extractFilePath(name);
      if ((!path.empty()) && (!isRoot(path)))
        path += dirsep;
      if (extractFileExtension(name) != "")
        name = path + newbasename + "." + extractFileExtension(name);
      else
        name = path + newbasename;
    }
    else
    {
      // this is for uniformity - it returns name in boost format
      // if name is "./dir" it returns ".\\dir" instead of "./dir"
      name = fs::path(name).string();
      if (!name.empty() && !isRoot(name))
        name = stripTrailingSlash(name);
    }
  }


//---------------------------------------------------------------------------
  void changePath(std::string &name, const std::string &newpath)
  {
    if (!newpath.empty())
    {
      std::string spath = fs::path(newpath).string();
//    if ((!isRoot(spath)) && (extractFileName(name) != ""))
//      spath += dirsep;
      if (extractFileName(name) != "")
        name = spath + ((spath[spath.length()-1] == dirsep[0]) ? std::string("") : dirsep) +  extractFileName(name);
      else
      {
        name = spath;
        if (!isRoot(name))
          name = stripTrailingSlash(spath);
      }
    }
    else
      // ???? Is this the behaviour we want ?
      name = extractFileName(name);
  }


//---------------------------------------------------------------------------
  bool readFile(const std::string &filename, std::string &s)
  {
	long n = fileSize(filename);
    if (n == -1)
      return false;
	FILE *fpIO = fopen(filename.c_str(), "rb");
	if (fpIO == NULL)
	  return false;
    s.clear();
    s.resize(n);
    //char *in = new char[n];
    long bytes = static_cast<long>(fread(&(s[0]), sizeof(char), n, fpIO));
	//delete in;
    fclose(fpIO);
    return (bytes == n);
  }

//---------------------------------------------------------------------------
  bool readFile(const std::string &filename, std::vector<std::string> &vs)
  {
	std::string s;
    bool res = readFile(filename, s);
    if (!res) return false;
    vs.clear();
    vs = split(s, "\r\n");
    if (vs.size() < 2)
    {
      //it may be a unix file, try to split on \n
      vs.clear();
      vs = split(s, "\n");
    }
    return true;
  }

  bool readFile(
    const std::string &filename,
    std::vector<std::vector<std::string> > &vvs,
    const std::string& split_at)
  {
    //BvG: can be made more efficient...
    std::vector<std::string> vs;
    readFile(filename, vs);
    vvs.clear();
    for (std::vector<std::string>::iterator it = vs.begin(); it != vs.end(); ++it)
    {
      std::vector<std::string> line;
      line = split(*it, split_at);
      vvs.push_back(std::vector<std::string>());
      for (std::vector<std::string>::iterator it2 = line.begin(); it2 != line.end(); ++it2)
      {
        if (!it2->empty()) vvs.back().push_back(*it2);
      }
    }
    return true;
  }

  bool writeFile(const std::string &filename, const std::string &s)
  {
    std::string path = extractFilePath(filename);
    createDirectory(path);
    std::ofstream f(filename.c_str());
    if (!f) return false;
    f << s;
    return true;
  }

  bool writeFile(const std::string &filename, const std::vector<std::string> &vs)
  {
    std::string path = extractFilePath(filename);
    createDirectory(path);
    std::ofstream f(filename.c_str());
    if (!f) return false;
    for (unsigned int i = 0; i < vs.size(); ++i) f << vs[i] << std::endl;
    return f.good();
  }

  bool writeFile(
    const std::string &filename,
    const std::vector<std::vector<std::string> > &vvs,
    const std::string &split
  )
  {
    std::string path = extractFilePath(filename);
    createDirectory(path);
    std::ofstream f(filename.c_str());
    if (!f) return false;
    for (unsigned int i = 0; i < vvs.size(); ++i)
    {
      for (unsigned int j = 0; j < vvs[i].size() - 1; ++j)
      {
        f << vvs[i][j] << split;
      }
      if (vvs[i].size() > 0) f << vvs[i].back();
      f  << std::endl;
    }
    return f.good();
  }

  bool equivalentPaths(const std::string path1, const std::string path2)
  {
    return fs::equivalent(fs::path(path1), fs::path(path2));
  }

  void cleanFileName(std::string &file)
  {
    if (file.size() < 2) return;
    bool networkpath = (file.substr(0, 2) == "\\\\");
    replaceAll(file, "\\\\", "\\");
    if (networkpath) file = std::string("\\") + file;
    if (file.substr(file.size() - 1, 1) == dirsep)
      file = file.substr(0, file.size() - 1);
  }

  void cleanDirName(std::string &dir)
  {
    cleanFileName(dir);
    dir += dirsep;
  }

  void getTempFile(std::string &filename, const std::string &prefix)
  {
#ifdef WIN32
    char buffer[MAX_PATH];
    char fnbuffer[MAX_PATH];
    GetTempPath(MAX_PATH, buffer);
    GetTempFileName(buffer, prefix.c_str(), 0, fnbuffer);
    filename = fnbuffer;
#else
    //  char* buffer = std::getenv("TEMP");
    char buffer[MAX_PATH];
    sprintf(buffer, "%sXXXXXX", prefix.c_str());
    // WARNING: UGLY. Better this function should return a file handler
    close(mkstemp(buffer));
    filename = std::string(buffer);
#endif

  }

  void getTempDir(std::string &dirname)
  {
#ifdef WIN32
    char buffer[MAX_PATH];
    int length = GetTempPath(MAX_PATH, buffer);

    if (length == 0)
    {
      dirname = "c:\\";
    }
    else
      dirname = buffer;
#else
    dirname = "/tmp/";
#endif

  }

  void getEmptyTempDir(std::string &dirname, const std::string &parent)
  {
#ifdef WIN32
    std::string temp(parent);
    if (temp.empty()) getTempDir(temp);
    srand(static_cast<unsigned int>(time(NULL)));
    do
    {
      do
      {
        int i = rand();
        dirname = temp + std::to_string(i);
      }
      while (dirExists(dirname));
      createDirectory(dirname);
    }
    while (!emptyDir(dirname));
#endif
  }

 std::vector<std::string> getWindowsDriveLetters()
  {
    std::vector<std::string> driveletters;
#ifdef WIN32
    DWORD bitmask = GetLogicalDrives();
    char driveletter = 'A';
    std::string empty;
    for (char i = 0; i < 32; i++)
    {
      if (bitmask & 0x1)
      {
        driveletters.push_back(empty + (char)(driveletter + i));
        empty = "";
      }
      bitmask >>= 1;
    }
#endif
    return driveletters;
  }

  std::string getDirSeparator()
  {
    return dirsep;
  }

  bool isComplete(const std::string &spath)
  {
    return fs::path(spath).is_absolute();
  }


} // end namespace
