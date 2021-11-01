//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>
#include <Half.hpp>

#include <mapbox/variant.hpp>

namespace armnnUtils
{

// Standard declaration of TContainer used by ArmNN
// Changes to this declaration constitute an api/abi break, new types should be added as a separate declaration and
// merged on the next planned api/abi update.
using TContainer = mapbox::util::variant<std::vector<float>,
                                         std::vector<int>,
                                         std::vector<uint8_t>,
                                         std::vector<int8_t>,
                                         std::vector<armnn::Half>>;

} // namespace armnnUtils
