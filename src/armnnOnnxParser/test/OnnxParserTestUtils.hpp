//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

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

} // namespace armnnUtils