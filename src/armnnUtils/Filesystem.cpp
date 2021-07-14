//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

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

}
}
