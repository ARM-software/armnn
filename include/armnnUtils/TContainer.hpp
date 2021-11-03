//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

#include <mapbox/variant.hpp>

namespace armnnUtils
{

// Standard definition of TContainer used by ArmNN, use this definition or add alternative definitions here instead of
// defining your own.
    using TContainer =
    mapbox::util::variant<std::vector<float>, std::vector<int>, std::vector<unsigned char>, std::vector<int8_t>>;

} // namespace armnnUtils
