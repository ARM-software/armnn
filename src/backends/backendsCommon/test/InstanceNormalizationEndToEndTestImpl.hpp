//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>

#include <vector>

void InstanceNormalizationNhwcEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends);

void InstanceNormalizationNchwEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends);

void InstanceNormalizationNhwcEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends);

void InstanceNormalizationNchwEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends);
