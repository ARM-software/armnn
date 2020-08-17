//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#if defined(_MSC_VER)
// ghc includes Windows.h directly, bringing in macros that we don't want (e.g. min/max).
// By including Windows.h ourselves first (with appropriate options), we prevent this.
#include <common/include/WindowsWrapper.hpp>
#endif
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

namespace armnnUtils
{
namespace Filesystem
{

/// Returns a path to a file in the system temporary folder. If the file existed it will be deleted.
fs::path NamedTempFile(const char* fileName);

}
}
