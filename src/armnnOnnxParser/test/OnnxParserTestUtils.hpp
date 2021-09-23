//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>
#include <vector>

namespace armnnUtils
{

std::string ConstructTensorShapeString(const std::vector<int>& shape);

std::string ConstructIntsAttribute(const std::string& name, const std::vector<int>& value);

} // namespace armnnUtils