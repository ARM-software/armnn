//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OutputHandler.hpp"

#include <armnn/backends/ITensorHandle.hpp>
#include <backendsCommon/WorkloadDataCollector.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

void OutputHandler::SetTensorInfo(const TensorInfo& tensorInfo)
{
    m_TensorInfo = tensorInfo;
    m_bTensorInfoSet = true;
}

void OutputHandler::CreateTensorHandles(const IWorkloadFactory& factory, const bool IsMemoryManaged)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    m_TensorHandle = factory.CreateTensorHandle(m_TensorInfo, IsMemoryManaged);
    ARMNN_NO_DEPRECATE_WARN_END
}

void OutputHandler::CreateTensorHandles(const ITensorHandleFactory& factory, const bool IsMemoryManaged)
{
    m_TensorHandle = factory.CreateTensorHandle(m_TensorInfo, IsMemoryManaged);
}

void OutputHandler::CollectWorkloadOutputs(WorkloadDataCollector& dataCollector) const
{
    dataCollector.Push(m_TensorHandle.get(), m_TensorInfo);
}

void OutputHandler::SetAllocatedData()
{
    // Set allocated data only once
    if (!m_AllocatedTensorHandle)
    {
       m_AllocatedTensorHandle = std::move(m_TensorHandle);
    }
}

} // namespace armnn
