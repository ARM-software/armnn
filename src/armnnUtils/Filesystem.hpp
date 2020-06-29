//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

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
