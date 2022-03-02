//
// Copyright © 2022 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{

namespace pipe
{

enum class ProfilingState
{
    Uninitialised,
    NotConnected,
    WaitingForAck,
    Active
};

} // namespace pipe

} // namespace arm
