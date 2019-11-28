//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OutputHandler.hpp"

#include <armnn/backends/ITensorHandle.hpp>
#include <backendsCommon/WorkloadDataCollector.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <boost/assert.hpp>

namespace armnn
{

void OutputHandler::SetTensorInfo(const TensorInfo& tensorInfo)
{
    m_TensorInfo = tensorInfo;
    m_bTensorInfoSet = true;
}

void OutputHandler::CreateTensorHandles(const IWorkloadFactory& factory, const bool IsMemoryManaged)
{
    m_TensorHandle = factory.CreateTensorHandle(m_TensorInfo, IsMemoryManaged);
}

void OutputHandler::CreateTensorHandles(const ITensorHandleFactory& factory, const bool IsMemoryManaged)
{
    m_TensorHandle = factory.CreateTensorHandle(m_TensorInfo, IsMemoryManaged);
}

void OutputHandler::CollectWorkloadOutputs(WorkloadDataCollector& dataCollector) const
{
    dataCollector.Push(m_TensorHandle.get(), m_TensorInfo);
}

} // namespace armnn
