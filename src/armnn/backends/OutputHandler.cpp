//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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

void OutputHandler::CollectWorkloadOutputs(WorkloadDataCollector& dataCollector) const
{
    dataCollector.Push(m_TensorHandle.get(), m_TensorInfo);
}

} // namespace armnn
