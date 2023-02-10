//
// Copyright © 2020,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#if !defined(ARMNN_DISABLE_FILESYSTEM)

#if defined(_MSC_VER)
// ghc includes Windows.h directly, bringing in macros that we don't want (e.g. min/max).
// By including Windows.h ourselves first (with appropriate options), we prevent this.
#include <common/include/WindowsWrapper.hpp>
#endif
#include <ghc/filesystem.hpp>
#include <string>

namespace fs = ghc::filesystem;

namespace armnnUtils
{
namespace Filesystem
{

using FileContents = std::string;

/// Returns a path to a file in the system temporary folder. If the file existed it will be deleted.
fs::path NamedTempFile(const char* fileName);

/// Returns full path to temporary folder
std::string CreateDirectory(std::string sPath);

FileContents ReadFileContentsIntoString(const std::string& path);

void RemoveDirectoryAndContents(const std::string& path);

} // namespace armnnUtils
} // namespace Filesystem

#endif // !defined(ARMNN_DISABLE_FILESYSTEM)
