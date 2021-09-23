//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OnnxParserTestUtils.hpp"

#include <fmt/format.h>

namespace armnnUtils
{

std::string ConstructTensorShapeString(const std::vector<int>& shape)
{
    std::string shapeStr;
    for (int i : shape)
    {
        shapeStr = fmt::format("{} dim {{ dim_value: {} }}", shapeStr, i);
    }
    return shapeStr;
}

std::string ConstructIntsAttribute(const std::string& name,
                                   const std::vector<int>& values)
{
    std::string attrString = fmt::format("attribute {{ name: '{}'", name);;
    for (int i : values)
    {
        attrString = fmt::format(" {} ints: {}", attrString, i);
    }
    attrString = fmt::format(" {} type: INTS }}", attrString);
    return attrString;
}

} // namespace armnnUtils