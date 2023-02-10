//
// Copyright © 2020,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#if !defined(ARMNN_DISABLE_FILESYSTEM)

#include <armnn/Exceptions.hpp>
#include <armnnUtils/Filesystem.hpp>

namespace armnnUtils
{
namespace Filesystem
{

/**
 * @brief Construct a temporary file name.
 *
 * Given a specified file name construct a path to that file in the
 * system temporary directory. If the file already exists it is deleted. This
 * could throw filesystem_error exceptions.
 *
 * @param fileName the file name required in the temporary directory.
 * @return path consisting of system temporary directory and file name.
 */
fs::path NamedTempFile(const char* fileName)
{
    fs::path tmpDir = fs::temp_directory_path();
    fs::path namedTempFile{tmpDir / fileName};
    if (fs::exists(namedTempFile))
    {
        fs::remove(namedTempFile);
    }
    return namedTempFile;
}

/**
 * @brief Construct a temporary directory
 *
 * Given a specified directory name construct a path in the
 * system temporary directory. If the directory already exists, it is deleted,
 * otherwise create it. This could throw filesystem_error exceptions.
 *
 * @param path is the path required in the temporary directory.
 * @return path consisting of system temporary directory.
 * @throws RuntimeException if the directory cannot be created or exists but cannot be removed.
 */
std::string CreateDirectory(std::string path)
{
    // This line is very unlikely to throw an exception.
    fs::path tmpDir = fs::temp_directory_path();
    std::string full_path = tmpDir.generic_string() + path;
    // This could throw a file permission exception.
    RemoveDirectoryAndContents(full_path);
#if defined(_WIN32)
    result = _mkdir(full_path.c_str()); // can be used on Windows
    armnn::ConditionalThrow<armnn::RuntimeException>((result == 0), "Was unable to create temporary directory");
#else
    try
    {
        if(!fs::create_directory(full_path))
        {
            throw armnn::RuntimeException("Unable to create directory: " + full_path);
        }
    }
    catch (const std::system_error& e)
    {
        std::string error = "Unable to create directory. Reason: ";
        error.append(e.what());
        throw armnn::RuntimeException(error);
    }
#endif

    return full_path + "/";
}

/**
 * @brief Remove a directory and its contents.
 *
 * Given a directory path delete it's contents and the directory. If the specified directory doesn't exist this
 * does nothing. If any item cannot be removed this will throw a RuntimeException.
 *
 * @param full_path
 */
void RemoveDirectoryAndContents(const std::string& path)
{
    if (fs::exists(path))
    {
        try
        {
            // This could throw an exception on a multi-user system.
            fs::remove_all(path);
        }
        catch (const std::system_error& e)
        {
            std::string error = "Directory exists and cannot be removed. Reason: ";
            error.append(e.what());
            throw armnn::RuntimeException(error);
        }
    }
}

FileContents ReadFileContentsIntoString(const std::string& path) {
    if (!fs::exists(path))
    {
        throw armnn::RuntimeException("Path does not exist: " + path);
    }
    std::ifstream input_file(path);
    armnn::ConditionalThrow<armnn::RuntimeException>((input_file.is_open()), "Could not read file contents");
    return FileContents((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
}

} // namespace armnnUtils
} // namespace Filesystem

#endif // !defined(ARMNN_DISABLE_FILESYSTEM)
