//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cassert>

namespace armnn
{

#ifndef NDEBUG
#   define ARMNN_ASSERT(COND) assert(COND)
#   define ARMNN_ASSERT_MSG(COND, MSG) assert((COND) && MSG)
#else
#   define ARMNN_ASSERT(COND)
#   define ARMNN_ASSERT_MSG(COND, MSG)
#endif

} //namespace armnn