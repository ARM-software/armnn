//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>

#include <vector>

void LogSoftmaxEndToEndTest(const std::vector<armnn::BackendId>& defaultBackends);
