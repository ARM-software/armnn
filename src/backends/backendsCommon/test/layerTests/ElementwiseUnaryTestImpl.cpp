//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseUnaryTestImpl.hpp"

std::unique_ptr<armnn::IWorkload> CreateWorkload(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::ElementwiseUnaryQueueDescriptor& descriptor)
{
    return workloadFactory.CreateWorkload(armnn::LayerType::ElementwiseUnary, descriptor, info);
}