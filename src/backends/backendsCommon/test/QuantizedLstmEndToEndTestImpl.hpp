//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>

#include <vector>

void QuantizedLstmEndToEnd(const std::vector<armnn::BackendId>& backends);
