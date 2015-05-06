#ifndef __DIAGCVR_FILETOOLS_H__
#define __DIAGCVR_FILETOOLS_H__

#include <config/pathology_config.h>
#include <string>
#include <vector>

#ifdef WIN32
#include <windows.h>
#include <tchar.h>
#include <shellapi.h>
#endif

namespace core
{

// FUNCTION GROUP: file utility functions:

/////////////////////
// Returns true iff file exists (it is not checked if the file can be opened)
  bool EXPORT_CORE fileExists(const std::string &name);

/////////////////////
// Returns true iff the directory exists
  bool EXPORT_CORE dirExists(const std::string &name);

//////////////////////
// Returns the size of the file in bytes.
// If the file does not exist or an error occurs, -1 is returned.
// Note: long int is just an int for the Borland compiler, so maximum file size is 2GB
  long int EXPORT_CORE fileSize(const std::string &name);

/////////////////////////
// Deletes the file, returns true iff the file existed and was successfully
// deleted.
  bool EXPORT_CORE deleteFile(const std::string &name);

/////////////////////////
// Deletes the directory, returns true iff the directory existed, is empty,
// and was successfully deleted.
  bool EXPORT_CORE deleteDir(const std::string &name, bool deleteNonEmpty = false);

#ifdef WIN32
  bool DeleteDirectory(LPCTSTR lpszDir, bool noRecycleBin = true);
#endif

/////////////////////////
// Returns true iff the directory exists and is empty.
  bool EXPORT_CORE emptyDir(const std::string &name);

/////////////////////////
// Copies the file, returns true if the source file exists and is not empty, and
// the target file does not exist and the copy is successful.
// If the target directory does exist, it is created.
// Flag overwrite to overwrite an existing target file
  bool EXPORT_CORE copyFile(
    const std::string &source,
    const std::string &target,
    bool overwrite = false,
    bool copyAttributes = true
    );

///////////////////////////
// Copies all files in a directory. Returns true if the source path exists and
// the target directory is empty and if copying was succesful.
// If the target directory does not exist it will be created.
// Specify name to only copy the files that match that pattern, this may
// include wildcards.
// If recurse is true, all subdirectories of path are processed as well. Source
// directory stucture is preserved.
// Flag overwrite to overwrite existing files in the target directory.
  bool EXPORT_CORE copyDirectory(
    const std::string &source,
    const std::string &target,
    const std::string &name = "",
    bool recurse = false,
    bool overwrite = false,
    bool copyAttributes = true
    );


/////////////////////////
// Renames the file, returns true iff the file existed and was successfully
// renamed. The target directory must exist, the target file should not exist.
  bool EXPORT_CORE renameFile(const std::string &source, const std::string &target);

///////////////////////
// Returns the path of the passed file name. Examples:
// File test.txt has an empty string as path<br>
// File c:\test.txt has path: c:\<br>
// File c:test.txt has path: c:<br>
// File \test.txt has path: \<br>
// File c:\temp\test.txt has path: c:\temp<br>
// File c:/temp/test.txt has path: c:\temp<br>
  std::string EXPORT_CORE extractFilePath(const std::string &name);

///////////////////////
// Returns the level above the current directory structure level
// If a root directory is given it is returned unchanged
// If a file name is attached to the directory given then the filePath
// to the file is returned
  std::string EXPORT_CORE upOneLevel(const std::string &name);

///////////////////////
// Returns the level above the current directory structure level
// If a root directory is given it is returned unchanged
// If a file name is attached to the directory given then the filePath
// to the file is returned
  std::string EXPORT_CORE upMultipleLevels(const std::string& name, unsigned int nrOfLevels);

///////////////////////
// Returns the undecorated name of the passed (full) file name.
// If the input is empty or ends with \ or / or : an empty string is returned.
  std::string EXPORT_CORE extractFileName(const std::string &name);

///////////////////////
// Returns the undecorated file name, as provided by extractFileName, but
// without the final extension. of the passed (full) file name. Examples:
// File test.txt returns test<br>
// File c:\test.txt returns test<br>
// File c:test.txt.bak returns test.txt<br>
// File \test returns test<br>
  std::string EXPORT_CORE extractBaseName(const std::string &name);

///////////////////////
// Returns the lowest level directory name
// Only works if "name" is a directory (not a file)
// Dir c:\level1\level2\level3 returns level3<br>
  std::string EXPORT_CORE extractLowestDirName(const std::string &name);

///////////////////////
// Returns the extension, if any, without the .
  std::string EXPORT_CORE extractFileExtension(const std::string &name);

////////////////////////
// Creates the directory, and all higher directories if necessary.
// Returns true iff the directory exists on exit.
  bool EXPORT_CORE createDirectory(const std::string &dir);

//////////////////////////
// Returns in v all files in path that match the pattern name, which may contain
// wildcards. If name is empty, all files are returned. If recurse is true, all
// subdirectories of path are processed as well.
  void EXPORT_CORE getFiles(
    const std::string &path,
    const std::string &name,
    std::vector<std::string> &v,
    bool recurse = false
  );

//////////////////////////
// Returns in v all subdirectories of path.
// If recurse is true, all subdirectories of path are processed as well.
  void EXPORT_CORE getSubdirectories(
    const std::string &path,
    std::vector<std::string> &v,
    bool recurse = false
  );

#ifndef LINUX
  void EXPORT_CORE getSubdirectoriesWindows(
    const std::string &thepath,
    std::vector<std::string> &v,
    bool recurse);
#endif

//////////////////////////
// Creates the directory, including any subdirectory is needed.
// path can be a filename or a path name (extractdirectoryName is called first).
// Returns true if the dierctory exists on exit, false otherwise.
  bool EXPORT_CORE createDirectory(const std::string &path);

//////////////////////////
// Returns the name of the root directory in the specified path, or
// an empty string if the path is not complete. Examples:
// Path D:\dir1 returns D:<br>
// Path D:/dir1 returns returns D:<br>
// Path D: returns D:<br>
// Path //Hiawatha/bin returns //Hiawatha<br>
// Path \\Hiawatha\bin returns //Hiawatha<br>
// Path //Hiawatha/ returns //Hiawatha<br>
  std::string EXPORT_CORE rootName(const std::string &spath);

//////////////////////////
  /* Returns the path to pathToAlter starting from fixedPath
     If the items cannot be related (different roots) then pathToAlter is returned. Examples:
     pathToAlter = c:\\mypath\otherpath\\test1\\a.txt, fixedPath = c:\\mypath\\diffpath\\
                           returns ..\\otherpath\\test1\\a.txt
  */
  std::string EXPORT_CORE getPathRelativeToLocation(const std::string &pathToAlter, const std::string &fixedPath);

//////////////////////////
// Returns true if the given path is a path to the root and false otherwise.
// Examples:
// Path D: returns true<br>
// Path D:/ returns true<br>
// Path D:\ returns true<br>
// Path D:\dir returns false<br>
// Path //Hiawatha should return true<br> but returns false (bug in boost)(works in 1.35)
// Path \\Hiawatha should return true<br> but returns false (bug in boost)(works in 1.35)
// Path //Hiawatha/ returns true<br>
// Path //Hiawatha/dir returns false<br>
  bool EXPORT_CORE isRoot(const std::string &spath);

///////////////////////
// Returns true if the given path is a UNC path. This means it starts with \\.
  bool EXPORT_CORE isUNCPath(const std::string &spath);

//////////////////////////
// Returns true if the given path is a path to a directory and
// false if it is a path to a file.
// Examples:


// hello
// !!!@
  bool EXPORT_CORE isOnlyDirectory(const std::string &spath);

  bool EXPORT_CORE isOnlyDirectoryTmpKeelin(const std::string &spath);

//////////////////////////
// Returns the path to the current directory as maintained by the operating
// system.
// NOTE: Because the current path maintained by the operating system may be
// changed at any moment (by a third-party or system library function, or by
// another thread) it is not safe to use this function repetitively in a program.
// If your program depends on the current directory, it is good to save the return
// value of currentDirPath() immediately upon entering main(), and use this
// variable from there on.
  std::string EXPORT_CORE currentDirPath();

//////////////////////////
// Creates a complete path from base and path. (A complete path is a path containing
// the root directory.) Usually, base represents a path to the current directory,
// and path is a relative path.
// Examples:
// completePath( "dir1", "D:\\dir0" ) returns D:\dir0\dir1<br>
// completePath( "..", "D:\\dir0\\dir1" ) returns D:\dir0<br>
// completePath( "./", "D:\\dir0\\dir1" ) returns D:\dir0\dir1<br>
// completePath( "\\", "D:\\dir0\\dir1" ) returns D:\<br>
// completePath( "\\dir2", "D:\\dir0\\dir1" ) returns D:\dir2<br>
  std::string EXPORT_CORE completePath(const std::string &spath, const std::string &base);

/////////////////////////////
// Changes the extension of name. newextension should be the new extension, without the .
// Examples:
// changeExtension("c:\\test.txt","bak") changes input to c:\\test.bak
  void EXPORT_CORE changeExtension(std::string &name, const std::string &newextension);

/////////////////////////////
// Changes the basename. newbasename should be the new basename, without the .
// Examples:
// changeBaseName("c:\\test.txt","aap") changes input to c:\\aap.txt
  void EXPORT_CORE changeBaseName(std::string &name, const std::string &newbasename);

/////////////////////////////
// Changes the path. newpath should be the new path, and may or may not end with
// \\ or /
// Examples:
// changePath("c:\\test.txt","c:\temp") changes input to c:\temp\test.txt
  void EXPORT_CORE changePath(std::string &name, const std::string &newpath);

////////////////////////////
// Reads the contents of a text file into s. Returns true iff file exists and
// was successfully read.
  bool EXPORT_CORE readFile(const std::string &filename, std::string &s);

////////////////////////////
// Reads the contents of a text file into a vector of strings (one per line).
// Returns true iff file exists and was successfully read.
  bool EXPORT_CORE readFile(const std::string &filename, std::vector<std::string> &vs);

  bool EXPORT_CORE readFileTail(const std::string &filename, std::vector<std::string> &vs, int nBytesToRead = 2048);

////////////////////////////
// Reads the contents of a text file into a vector of vector of strings.
// The outer vector contains lines, the inner vector contains the elements of
// that line, split with the split character.
// Returns true iff file exists and was successfully read.
  bool EXPORT_CORE readFile(
    const std::string &filename,
    std::vector<std::vector<std::string> > &vvs,
    const std::string& split_at = " ");

////////////////////////////
// Writes s to a file. Returns true iff writing was successful.
  bool EXPORT_CORE writeFile(const std::string &filename, const std::string &s);

////////////////////////////
// Writes vs to a file, one string per line. Returns true iff writing was successful.
  bool EXPORT_CORE writeFile(const std::string &filename, const std::vector<std::string> &vs);

////////////////////////////
// Writes vvs to a file, one set of strings per line, each spearated by split.
// Returns true iff writing was successful.
  bool EXPORT_CORE writeFile(
    const std::string &filename,
    const std::vector<std::vector<std::string> > &vvs,
    const std::string &split = " "
  );

////////////////////////////
// Returns true if the two paths resolve to the same file or directory. (Useful
// for example when one of the paths is a network path and the other a local path).
  bool EXPORT_CORE equivalentPaths(const std::string path1,  const std::string path2);

//////////////////////////////
// 'Cleans' a filename, that is, double separators (two backslashes) are replaced
// with one, except for the first part of the path where it means that it is
// a different machine.
  void EXPORT_CORE cleanFileName(std::string &file);

//////////////////////////////
// 'Cleans' a directory name, similar to cleanFileName except that the final
// character is always a backslash
  void EXPORT_CORE cleanDirName(std::string &dir);

/////////////////////////////
// Returns the last date and time that the file was written to. This is the
// only type of datetime access provided by Boost Filesystem. If the file
// does not exist the function returns with all output set to -1.
  void EXPORT_CORE fileDateTime(
    const std::string &file,
    int &year,
    int &month,
    int &day,
    int &hour,
    int &min,
    int &sec
  );

/////////////////////////////
// Returns the last date and time that the file was written to in string
// format YYYYMMDDHHMMSS. Calls other fileDateTime function. Returns empty
// string on error.
  void EXPORT_CORE fileDateTime(
    const std::string &file, std::string &s);

/////////////////////////////
// Returns the current date and time in string format YYYYMMDDHHMMSS.
  void EXPORT_CORE getDateTime(std::string &s);

/////////////////////
// Windows only.
// Returns the name of a temp file ready to be used (the file is also created)
// The first three letters of the prefix string are prefixed to the temp file
// name.
  void EXPORT_CORE getTempFile(
    std::string &filename, // returned and a file with this name is created
    const std::string &prefix = std::string("pre")
  );

/////////////////////
// Windows only.
// Returns a temporary directory on the host
  void EXPORT_CORE getTempDir(std::string &dirname);

////////////////////
// Windows only.
// Returns a newly created, empty temporary directory on the host.
// Optionally the parent dir can be specified, otherwise geTempDir is used.
// May get stuck in an endless loop if it cannot create a directory in the
// parent directory.
  void EXPORT_CORE getEmptyTempDir(
    std::string &dirname, // empty dir, created and returned
    const std::string &parent = std::string()
  );

////////////////////
// Windows only.
// Returns a vector with string objects each representing an available windows
// drive. This function can be used to determine which drives are
// (un)available on a windows computer.
  std::vector<std::string> EXPORT_CORE getWindowsDriveLetters();

  std::string EXPORT_CORE stripTrailingSlash(std::string str);

  std::string EXPORT_CORE getDirSeparator();

  bool EXPORT_CORE isComplete(const std::string &spath);

  std::string EXPORT_CORE uniformSlashes(const std::string &path);

}

#endif
