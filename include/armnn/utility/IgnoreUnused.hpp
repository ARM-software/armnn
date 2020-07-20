//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

// Utility function to selectively silence unused variable compiler warnings

template<typename ... Ts>
inline void IgnoreUnused(Ts&&...){}

} //namespace armnn