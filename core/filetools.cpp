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
const std::string dirsep("/");
#define MAX_PATH PATH_MAX
#endif

#define BOOST_FILESYSTEM_VERSION 3

#include "stringconversion.h"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/convenience.hpp"
#include "boost/version.hpp"
#include "boost/regex.hpp"
#include <stdio.h>
#include <fcntl.h>
#include <iostream>
#include <cstdlib>
#include <fstream>

using namespace boost::filesystem;
using namespace std;

namespace core
{

// Implementation of these function uses boost filesystem library
// when possible. The reason for providing these functions is that
// it is easier to remember such convenience functions than the syntax
// of the Boost filesystem library.

  bool fileExists(const std::string &name)
  {
    try
    {
      path p(name);
      if (!exists(p)) return false;
      if (is_directory(p)) return false;
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
      path p(name);
      bool ex = exists(p);
      if (!ex) return false;
      ex = is_directory(p);
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
	long int length = file_size(name);
	/*
	FILE *handle = fopen(name.c_str(), "rb");

	if (handle == NULL) return -1;
	fseek(handle, 0, SEEK_END);
	long int length = ftell(handle); //determine filelength
	fclose(handle);
	*/
    return length;
  }
//---------------------------------------------------------------------------
  bool deleteFile(const std::string &name)
  {
    if (!fileExists(name)) return false;
    try
    {
      remove(path(uniformSlashes(name)));
    }
    catch (...)
    {
      return false;
    }
    return true;
  }

  string uniformSlashes(const string &path)
  {
    string localpath = path;
#ifdef WIN32
    replace(localpath.begin(), localpath.end(), '/', '\\');
#else
    replace(localpath.begin(), localpath.end(), '\\', '/');
#endif
    return localpath;
  }

//---------------------------------------------------------------------------

  /*
    #ifdef WIN32
    #include <windows.h>
    #include <tchar.h>
    #include <shellapi.h>

    bool deleteDir(const std::string &name, bool deleteNonEmpty)
    {
      if (!dirExists(name)) return false;
      if(!deleteNonEmpty)
        if (!emptyDir(name)) return false;
      return DeleteDirectory(path(name).string().c_str());
    }

    bool DeleteDirectory(LPCTSTR lpszDir, bool noRecycleBin)
    {
      int len = _tcslen(lpszDir);
      TCHAR *pszFrom = new TCHAR[len+2];
      _tcscpy(pszFrom, lpszDir);
      pszFrom[len] = 0;
      pszFrom[len+1] = 0;

      SHFILEOPSTRUCT fileop;
      fileop.hwnd   = NULL;    // no status display
      fileop.wFunc  = FO_DELETE;  // delete operation
      fileop.pFrom  = pszFrom;  // source file name as double null terminated string
      fileop.pTo    = NULL;    // no destination needed
      fileop.fFlags = FOF_NOCONFIRMATION | FOF_SILENT;  // do not prompt the user

      if (!noRecycleBin)
        fileop.fFlags |= FOF_ALLOWUNDO;

      fileop.fAnyOperationsAborted = FALSE;
      fileop.lpszProgressTitle     = NULL;
      fileop.hNameMappings         = NULL;

      int ret = SHFileOperation(&fileop);
      delete [] pszFrom;
      return (ret == 0);
    }

    #else
  */


  bool deleteDir(const std::string &name, bool deleteNonEmpty)
  {
    if (!dirExists(name)) return false;
    if (!deleteNonEmpty)
      if (!emptyDir(name)) return false;
    return 0 != remove_all(path(uniformSlashes(name)));
//    return remove(path(name));
  }
//#endif



//---------------------------------------------------------------------------
  bool emptyDir(const std::string &name)
  {
    if (!dirExists(name)) return false;
    return boost::filesystem::is_empty(path(name));
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
      ifstream ifs(source.c_str());
      std::ofstream ofs(target.c_str());
      ofs << ifs.rdbuf();
      ofs.close();
      ifs.close();
    }
    else
    {
      copy_file(path(source), path(target));
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
    string s(source), t(target);
    cleanFileName(s);
    cleanFileName(t);
    if (!fileExists(s)) return false;
    if (fileExists(t)) return false;
    string dir = extractFilePath(t);
    createDirectory(dir);
    //BvG: if target is on a different drive, source will also be deleted
    rename(path(s), path(t));
    return fileExists(t);
  }

//---------------------------------------------------------------------------
  std::string upOneLevel(const std::string &name)
  {
    if (isRoot(name))
      return name;

    string s(name);
    if (isOnlyDirectory(name))
      s = stripTrailingSlash(s);

    path p(s);
    return p.branch_path().string();
  }

  std::string upMultipleLevels(const std::string &name, unsigned int nrOfLevels)
  {
    if (isRoot(name))
      return name;

    string s(name);
    if (isOnlyDirectory(name))
      s = stripTrailingSlash(s);

    path p(s);
    if (nrOfLevels == 1)
      return p.branch_path().string();
    else
      return upMultipleLevels(p.branch_path().string(), nrOfLevels - 1);
  }

//---------------------------------------------------------------------------
  std::string extractFilePath(const std::string &name)
  {
    string s(name);
    if (isOnlyDirectory(s))
    {
      path p(s);
      if (isRoot(name))
        return p.string();
      else
        return stripTrailingSlash(string(p.string()));
    }
    else
    {
      cleanFileName(s);
      path p(s);
      if (isRoot(name)) // CM: needed for "\\\\machine\\" see testFileTools.cpp
        return p.string();
      return string(p.parent_path().string());
    }
  }

//---------------------------------------------------------------------------
  std::string extractFileName(const std::string &name)
  {
    string s(name);
    if (isOnlyDirectory(s)) return string();
    cleanFileName(s);
    path p(s);
    if (isRoot(name)) return string();
    return p.leaf().string();
  }

//---------------------------------------------------------------------------
  std::string extractLowestDirName(const std::string &name)
  {
    // Note that this function now ASSUMES!! a directory is given as input
    string s(name);
    // CM: if both lines below are use, it makes no sense, cleanDir adds "\\" and thus isOnlyDirectory will always succeed
//  cleanDirName(s);
//  if (! isOnlyDirectory(s)) return string();
    s = stripTrailingSlash(s);
    path p(s);
    return p.leaf().string();
  }


//---------------------------------------------------------------------------
  std::string extractBaseName(const std::string &name)
  {
    string s(name);
    if (isOnlyDirectory(name)) return string();
    cleanFileName(s);
    string res = extractFileName(s);
    size_t pos = res.rfind(".");
    if (pos != string::npos) res.resize(pos);
    return res;
  }


//---------------------------------------------------------------------------
  string extractFileExtension(const std::string &name)
  {
    string s = extractFileName(name);
    size_t pos = s.rfind('.');
    if (pos != string::npos) return string(s.begin() + pos + 1, s.end());
    return string();
  }

//---------------------------------------------------------------------------
  bool createDirectory(const std::string &dir)
  {
    string s(dir);
    cleanDirName(s);
    try
    {
      path p(s);
      create_directories(p);
    }
    catch ( ... )
    {
    }
    return dirExists(s);
  }

//---------------------------------------------------------------------------
  string getPathRelativeToLocation(const std::string &_pathToAlter, const std::string &_fixedPath)
  {
//Make copies so we can change them for processing
    string pathToAlter = _pathToAlter;
    string fixedPath = _fixedPath;

//if a relative path cannot be written
    if (! (rootName(pathToAlter)).compare(rootName(fixedPath)) == 0 )
      return pathToAlter;
//if the two paths are the same
    if ( pathToAlter.compare(fixedPath) == 0)
      return string(".");

//if the path ends with a "\" then cut it off
    if (pathToAlter.find_last_of(dirsep) == pathToAlter.length() - 1)
      pathToAlter = pathToAlter.substr(0, pathToAlter.length() - 1);
    if (fixedPath.find_last_of(dirsep) == fixedPath.length() - 1)
      fixedPath = fixedPath.substr(0, fixedPath.length() - 1);


    string relativePath;
//If the entire fixed path is contained within the path to be given
    if ( pathToAlter.length() >= fixedPath.length()
         && pathToAlter.substr(0, fixedPath.length()).compare(fixedPath) == 0 )
    {
      //Tben we just need "./" and to append the remainder (non-matched part) of pathToAlter
      relativePath = string(".") + dirsep + pathToAlter.substr(fixedPath.length() + 1, pathToAlter.length() - fixedPath.length());
    }
//Otherwise we need at least one "../" step in the relative path
    else
    {
      relativePath = string("..") + dirsep;
      //step up a level in the fixed path and see if it is now contained within the pathToAlter
      string choppedFixed = upOneLevel(fixedPath);
      //As long as the "choppedFixed" is not contained in pathToAlter
      while ( pathToAlter.substr(0, choppedFixed.length()).compare(choppedFixed) != 0)
      {
        //Keep stepping up one level and appending "../" to relative path
        relativePath = relativePath + string("..") + dirsep;
        choppedFixed = upOneLevel(choppedFixed);
      }

      //if choppedFixed ends with a "\" then cut it off (e.g. if it is C:\ then '\' always appended)
      if (choppedFixed.find_last_of(dirsep) == choppedFixed.length() - 1)
        choppedFixed = choppedFixed.substr(0, choppedFixed.length() - 1);

      //A string to hold the path that we need to append to the "../../../" we made
      string endOfRelPath = string("");

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
    path p(spath);
    return p.root_name().string();
  }

  bool isRoot(const std::string &spath)
  {
    if (spath.empty()) return false;
    path p(spath);
    string s1 = p.string();
    string s2 = p.root_path().string();
    return (s1 == s2);
  }

  bool isUNCPath(const std::string &spath)
  {
    string s = rootName(spath);
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

    string s(spath);
    cleanDirName(s);
    try
    {
      path p(s);
      if (p.leaf() == p.root_name())
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
      path mypath(spath);
      if (is_directory(mypath))
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
    path p = current_path();
    return p.string();
  }

//---------------------------------------------------------------------------

  std::string stripTrailingSlash(std::string strPath)
  {
#if (__CODEGEARC__ >= 0x610)
    char lastchar = strPath[strPath.size()-1];
    if (lastchar == '\\')
    {
      strPath = strPath.substr(0, strPath.size() - 1);
    }
    else
    {
      std::string x = strPath.substr(strPath.size() - 2, 2);
      if (x == "\\.")
      {
        strPath = strPath.substr(0, strPath.size() - 2);
      }
    }
#endif
#ifdef _MSC_VER
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
#endif
#ifndef WIN32
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
    path pp(cpath);
    if (pp.is_complete())
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
    path pb(s);

    path cp;
    if (! pp.empty())
    {
      if ((spath == "\\" || spath == "/") && pb.has_root_path())
        return pb.root_path().string();
      else
        cp = complete(pp, pb);
    }
    else cp = pb;

    cp.normalize();

    /*
    // If spath = "./dir1/dir2" and base = "d:/dir0", the complete function
    // then gives the following path "d:/dir0/./dir1/dir2". Because such a path
    // is not accepted by some applications (the net share command, for example,
    // does not recognise it) we remove below any "." component in the path cp.
    // This bugs was fixed in a newer version of boost - a function normalize()
    // was introduced which presumably does the above.

    vector<string> v;
    path p = cp;
    path p1;
    while (!p.string().empty())
    {
     p1 = p;
     if (p.leaf() != ".") v.push_back(p.leaf());
     p = p.branch_path();
    }

    cp = path();
    for (int i = v.size()-1; i >= 0; --i)
     cp /= path(v[i]);

    // cp.normalize();
    // when updating to Boost 1.32 this function could replace the above code
    */

    return stripTrailingSlash(cp.string());
  }

//---------------------------------------------------------------------------
//MN: WARNING, THIS FUNCTION HAS BEEN OBSERVED TO GENERATE FILESYSTEM EXCEPTIONS
//    IN THE WILD.
  void getFiles(
    const string &thepath,
    const string &name,
    vector<string> &v,
    bool recurse)
  {
    v.clear();
    string pa(thepath);
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

    vector<path> vp;
    vp.push_back(path(pa));
    unsigned int i = 0;

    //to create a proper regular expression, any non literal should be preceded by \:
    //All characters are literals except: ".", "|", "*", "?", "+", "(", ")", "{", "}",
    //"[", "]", "^", "$" and "\".
    //some of these characters are not allowed in filenames, ...
    //and * should be replaced by .* and ? should be replaced by .?
    string filename = name;
    if (filename.empty()) filename = "*";
    replaceAll(filename, ".", "\\.");
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
    boost::regex e(filename, boost::regbase::normal | boost::regbase::icase);

    while (i < vp.size())
    {
      //now we collect all files in the directory and check if they match the pattern
      directory_iterator end_itr; // default construction yields past-the-end
      string p = vp[i].string();
#ifdef WIN32
      if (p[p.size()-1] != '\\') p += "\\";
#else
      if (p[p.size()-1] != '/') p += "/";
#endif
      for (directory_iterator itr(vp[i]); itr != end_itr; ++itr)
      {
        //BvG: the call to is_directory could raise an exception if
        //the directory is not accessible
        bool b;
        try
        {
          b = is_directory(*itr);
        }
        catch ( ... )
        {
          // handler for any C++ exception
          b = true; // so we skip this one
          //libReport(eError,"Error in getFiles: is_directory %s\n",itr->string().c_str());
        }
        if ( !b )
        {
          string f = itr->path().filename().string();
          if (regex_match(f, e)) v.push_back(p + f);
        }
        else if (recurse)
        {

#if BOOST_VERSION <= 103301
          string subdir = itr->string() + dirsep;
#else
          string subdir = itr->path().string() + dirsep;
#endif
          vp.push_back(path(subdir));
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
    const string &thepath,
    vector<string> &v,
    bool recurse)
  {
    v.clear();
    string pa = thepath;
    vector<path> vp;
    path p(pa);
    vp.push_back(p);
    v.push_back(p.string());

    unsigned int i = 0;
    while (i < vp.size())
    {
      //now we collect all subdirectories of path vp[i]
      directory_iterator end_itr; // default construction yields past-the-end
      for (directory_iterator itr(vp[i]); itr != end_itr; ++itr)
      {
        //BvG: the call to is_directory could raise an exception if
        //the directory is not accessible
        bool b;
        try
        {
          b = is_directory(*itr);
        }
        catch ( ... )
        {
          // handler for any C++ exception
          b = false; // so we skip this one
          //libReport(eError,"Error in getSubdirectories: is_directory %s\n",itr->string().c_str());
        }
        if (b)
        {
#if BOOST_VERSION <= 103301
          string subdir = itr->string();
#else
          string subdir = itr->path().string();
#endif
          v.push_back(subdir);
          if (recurse)
            vp.push_back(path(subdir));
        }
      }
      ++i; // move to the next directory to process
    } // while (i<vp.size())
  }

#ifdef WIN32
  void getSubdirectoriesWindows(
    const string &thepath,
    vector<string> &v,
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
      string path = extractFilePath(name);
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
      name = path(name).string();
      if (!name.empty() && !isRoot(name))
        name = stripTrailingSlash(name);
    }
  }


//---------------------------------------------------------------------------
  void changeBaseName(std::string &name, const std::string &newbasename)
  {
    if (extractBaseName(name) != "")
    {
      string path =  extractFilePath(name);
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
      name = path(name).string();
      if (!name.empty() && !isRoot(name))
        name = stripTrailingSlash(name);
    }
  }


//---------------------------------------------------------------------------
  void changePath(std::string &name, const std::string &newpath)
  {
    if (!newpath.empty())
    {
      string spath = path(newpath).string();
//    if ((!isRoot(spath)) && (extractFileName(name) != ""))
//      spath += dirsep;
      if (extractFileName(name) != "")
        name = spath + ((spath[spath.length()-1] == dirsep[0]) ? string("") : dirsep) +  extractFileName(name);
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
    long bytes = fread(&(s[0]), sizeof(char), n, fpIO);
	//delete in;
    fclose(fpIO);
    return (bytes == n);
  }

//---------------------------------------------------------------------------
  bool readFile(const std::string &filename, std::vector<std::string> &vs)
  {
	string s;
    bool res = readFile(filename, s);
    if (!res) return false;
    vs.clear();
    split(s, vs, "\r\n");
    if (vs.size() < 2)
    {
      //it may be a unix file, try to split on \n
      vs.clear();
      split(s, vs, "\n");
    }
    return true;
  }

  bool readFileTail(const std::string &filename, std::vector<std::string> &vs, int nBytesToRead)
  {
#ifdef WIN32
    //get file size
   // HANDLE h = CreateFile(filename.c_str(), GENERIC_READ, FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	//DWORD size = GetFileSize(h, NULL);
    //CloseHandle(h);
	//
	 // stringstream ss;
	  string tail;
	FILE *pFile;
	pFile = fopen(filename.c_str(), "rb");
	if(!pFile) return false;
	//fseek(handle, 0, SEEK_END);
    //long int length = ftell(handle); /* determine filelength */
   //	int nBytesToRead = 2048; //size > 2048 ? 2048 : size;
	if (!fseek(pFile, -nBytesToRead, SEEK_END))
	{
	/*
	  int c;
	  do
	  {
		c = fgetc (pFile);
		ss << char(c);
	  }
	  while (c != EOF);
	  */
	  tail.resize(nBytesToRead);
	  long bytes = fread(&(tail[0]), sizeof(char), nBytesToRead, pFile);
	
	}
	else //smaller than nBytesToRead
	{
	  fclose(pFile);
	  return readFile(filename, vs);
	}
	  fclose(pFile);


	//
	//string tail = ss.str();
   //	libReport(eError, tail.c_str());
	vs.clear();
	split(tail, vs, "\r\n");
    if (vs.size() < 2)
    {
      //it may be a unix file, try to split on \n
      vs.clear();
      split(tail, vs, "\n");
    }
#else
	return readFile(filename, vs);
#endif
    return true;
  }

  bool readFile(
    const std::string &filename,
    std::vector<std::vector<std::string> > &vvs,
    const std::string& split_at)
  {
    //BvG: can be made more efficient...
    vector<string> vs;
    readFile(filename, vs);
    vvs.clear();
    for (vector<string>::iterator it = vs.begin(); it != vs.end(); ++it)
    {
      vector<string> line;
      split(*it, line, split_at);
      vvs.push_back(vector<string>());
      for (vector<string>::iterator it2 = line.begin(); it2 != line.end(); ++it2)
      {
        if (!it2->empty()) vvs.back().push_back(*it2);
      }
    }
    return true;
  }

  bool writeFile(const std::string &filename, const std::string &s)
  {
    string path = extractFilePath(filename);
    createDirectory(path);
    ofstream f(filename.c_str());
    if (!f) return false;
    f << s;
    return true;
  }

  bool writeFile(const std::string &filename, const std::vector<std::string> &vs)
  {
    string path = extractFilePath(filename);
    createDirectory(path);
    ofstream f(filename.c_str());
    if (!f) return false;
    for (unsigned int i = 0; i < vs.size(); ++i) f << vs[i] << endl;
    return f.good();
  }

  bool writeFile(
    const std::string &filename,
    const std::vector<std::vector<std::string> > &vvs,
    const std::string &split
  )
  {
    string path = extractFilePath(filename);
    createDirectory(path);
    ofstream f(filename.c_str());
    if (!f) return false;
    for (unsigned int i = 0; i < vvs.size(); ++i)
    {
      for (unsigned int j = 0; j < vvs[i].size() - 1; ++j)
      {
        f << vvs[i][j] << split;
      }
      if (vvs[i].size() > 0) f << vvs[i].back();
      f  << endl;
    }
    return f.good();
  }

  bool equivalentPaths(const string path1, const string path2)
  {
    return equivalent(path(path1), path(path2));
  }

  void cleanFileName(std::string &file)
  {
    if (file.size() < 2) return;
    bool networkpath = (file.substr(0, 2) == "\\\\");
    replaceAll(file, "\\\\", "\\");
    if (networkpath) file = string("\\") + file;
    if (file.substr(file.size() - 1, 1) == dirsep)
      file = file.substr(0, file.size() - 1);
  }

  void cleanDirName(std::string &dir)
  {
    cleanFileName(dir);
    dir += dirsep;
  }

  void fileDateTime(
    const std::string &file,
    int &year,
    int &month,
    int &day,
    int &hour,
    int &min,
    int &sec
  )
  {
    year = month = day = hour = min = sec = -1;
    if (!fileExists(file)) return;
    path p(file);
    std::time_t t = last_write_time(p);

    struct tm *tb;
    // converts date/time to a structure
    tb = localtime(&t);
    //These quantities give the time on a 24-hour clock, day of month (1 to 31),
    //month (0 to 11), weekday (Sunday equals 0), year - 1900, day of year
    //(0 to 365), and a flag that is nonzero if the daylight saving time
    //conversion should be applied.
    sec = tb->tm_sec;
    min = tb->tm_min;
    hour = tb->tm_hour;
    day = tb->tm_mday;
    month = tb->tm_mon + 1;
    year = tb->tm_year + 1900;
  }

  void getDateTime(std::string &s)
  {
    std::time_t t = time(NULL);

    struct tm *tb;
    // converts date/time to a structure
    tb = localtime(&t);

    s = tostring(tb->tm_year + 1900);
    if (tb->tm_mon + 1 < 10) s += "0";
    s += tostring(tb->tm_mon + 1);
    if (tb->tm_mday < 10) s += "0";
    s += tostring(tb->tm_mday);
    if (tb->tm_hour < 10) s += "0";
    s += tostring(tb->tm_hour);
    if (tb->tm_min < 10) s += "0";
    s += tostring(tb->tm_min);
    if (tb->tm_sec < 10) s += "0";
    s += tostring(tb->tm_sec);
  }

  void fileDateTime(
    const std::string &file, string &s)
  {
    s.clear();
    int y, mo, d, h, mi, se;
    fileDateTime(file, y, mo, d, h, mi, se);
    if (y < 0) return;
    s = tostring(y);
    if (mo < 10) s += "0";
    s += tostring(mo);
    if (d < 10) s += "0";
    s += tostring(d);
    if (h < 10) s += "0";
    s += tostring(h);
    if (mi < 10) s += "0";
    s += tostring(mi);
    if (se < 10) s += "0";
    s += tostring(se);
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
    filename = string(buffer);
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
    string temp(parent);
    if (temp.empty()) getTempDir(temp);
    srand(static_cast<unsigned int>(time(NULL)));
    do
    {
      do
      {
        int i = rand();
        dirname = temp + tostring(i);
      }
      while (dirExists(dirname));
      createDirectory(dirname);
    }
    while (!emptyDir(dirname));
#endif
  }

  vector<string> getWindowsDriveLetters()
  {
    vector<string> driveletters;
#ifdef WIN32
    DWORD bitmask = GetLogicalDrives();
    char driveletter = 'A';
    string empty;
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
    return path(spath).is_complete();
  }


} // end namespace
