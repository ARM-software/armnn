//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>

void Connect(armnn::IConnectableLayer* from, armnn::IConnectableLayer* to, const armnn::TensorInfo& tensorInfo,
             unsigned int fromIndex = 0, unsigned int toIndex = 0);
