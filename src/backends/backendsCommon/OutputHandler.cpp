//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OutputHandler.hpp"

#include "ITensorHandle.hpp"
#include "WorkloadDataCollector.hpp"

#include <backendsCommon/WorkloadFactory.hpp>

#include <boost/assert.hpp>
#include <boost/log/trivial.hpp>

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

void OutputHandler::CreateTensorHandles(const ITensorHandleFactory& factory)
{
    m_TensorHandle = factory.CreateTensorHandle(m_TensorInfo);
}

void OutputHandler::CollectWorkloadOutputs(WorkloadDataCollector& dataCollector) const
{
    dataCollector.Push(m_TensorHandle.get(), m_TensorInfo);
}

} // namespace armnn
