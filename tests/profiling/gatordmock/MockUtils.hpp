//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <EncodeVersion.hpp>
#include <ProfilingUtils.hpp>

namespace armnn
{

namespace gatordmock
{

std::string CentreAlignFormatting(const std::string& stringToPass, const int spacingWidth);

std::string GetStringNameFromBuffer(const unsigned char *const data, uint32_t offset);

bool IsValidChar(unsigned char c);

} // gatordmock

} // armnn
