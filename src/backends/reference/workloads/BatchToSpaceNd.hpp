//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <DataLayoutIndexed.hpp>

namespace armnn
{

void BatchToSpaceNd(const armnnUtils::DataLayoutIndexed& dataLayout,
                    const TensorInfo& inputTensorInfo,
                    const TensorInfo& outputTensorInfo,
                    const std::vector<unsigned int>& blockShape,
                    const std::vector<std::pair<unsigned int, unsigned int>>& cropsData,
                    const float* inputData,
                    float* outputData);
} // namespace armnn
