//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cassert>

namespace arm
{

namespace pipe
{

#ifndef NDEBUG
#   define ARM_PIPE_ASSERT(COND) assert(COND)
#   define ARM_PIPE_ASSERT_MSG(COND, MSG) assert((COND) && MSG)
#else
#   define ARM_PIPE_ASSERT(COND)
#   define ARM_PIPE_ASSERT_MSG(COND, MSG)
#endif
} // namespace pipe
} //namespace arm