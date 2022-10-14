//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#if !defined(ARMNN_DISABLE_FILESYSTEM)

#include <armnnUtils/Filesystem.hpp>
#include "armnn/Exceptions.hpp"

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
 */
std::string CreateDirectory(std::string path)
{
    fs::path tmpDir = fs::temp_directory_path();
    mode_t permissions = 0733;
    int result = 0;

    std::string full_path = tmpDir.generic_string() + path;
    if (fs::exists(full_path))
    {
        fs::remove_all(full_path);
    }

#if defined(_WIN32)
    result = _mkdir(full_path.c_str()); // can be used on Windows
    armnn::ConditionalThrow<armnn::RuntimeException>((result == 0), "Was unable to create temporary directory");
#else
    result = mkdir(full_path.c_str(), permissions);
    armnn::ConditionalThrow<armnn::RuntimeException>((result == 0), "Was unable to create temporary directory");
#endif

    return full_path + "/";
}

FileContents ReadFileContentsIntoString(const std::string path) {
    std::ifstream input_file(path);
    armnn::ConditionalThrow<armnn::RuntimeException>((input_file.is_open()), "Could not read file contents");
    return FileContents((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
}

} // namespace armnnUtils
} // namespace Filesystem

#endif // !defined(ARMNN_DISABLE_FILESYSTEM)
