//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{

namespace pipe
{
// Static constants describing ArmNN's counter UID's
static const uint16_t NETWORK_LOADS         = 0;
static const uint16_t NETWORK_UNLOADS       = 1;
static const uint16_t REGISTERED_BACKENDS   = 2;
static const uint16_t UNREGISTERED_BACKENDS = 3;
static const uint16_t INFERENCES_RUN        = 4;
static const uint16_t MAX_ARMNN_COUNTER     = INFERENCES_RUN;
} // namespace pipe

} // namespace arm
