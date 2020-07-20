//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{

namespace pipe
{
// Utility function to selectively silence unused variable compiler warnings

template<typename ... Ts>
inline void IgnoreUnused(Ts&&...){}
} //namespace pipe
} //namespace arm