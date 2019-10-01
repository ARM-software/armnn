//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>

#include <vector>

void SpaceToDepthNhwcEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends);

void SpaceToDepthNchwEndToEndTest1(const std::vector<armnn::BackendId>& defaultBackends);

void SpaceToDepthNhwcEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends);

void SpaceToDepthNchwEndToEndTest2(const std::vector<armnn::BackendId>& defaultBackends);
