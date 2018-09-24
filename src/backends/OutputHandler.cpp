//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "OutputHandler.hpp"

#include <boost/assert.hpp>
#include <boost/log/trivial.hpp>

#include "backends/WorkloadFactory.hpp"
#include "backends/WorkloadDataCollector.hpp"
#include "backends/ITensorHandle.hpp"

namespace armnn
{

void OutputHandler::SetTensorInfo(const TensorInfo& tensorInfo)
{
    m_TensorInfo = tensorInfo;
    m_bTensorInfoSet = true;
}

void OutputHandler::CreateTensorHandles(const IWorkloadFactory& factory)
{
    m_TensorHandle = factory.CreateTensorHandle(m_TensorInfo);
}

void OutputHandler::CreateTensorHandles(const IWorkloadFactory& factory, DataLayout dataLayout)
{
    m_TensorHandle = factory.CreateTensorHandle(m_TensorInfo, dataLayout);
}

void OutputHandler::CollectWorkloadOutputs(WorkloadDataCollector& dataCollector) const
{
    dataCollector.Push(m_TensorHandle.get(), m_TensorInfo);
}

} // namespace armnn
