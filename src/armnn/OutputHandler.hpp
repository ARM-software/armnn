//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/ITensorHandle.hpp>
#include <armnn/backends/ITensorHandleFactory.hpp>

#include <armnn/Descriptors.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace armnn
{

class ITensorHandle;
class IWorkloadFactory;
class OutputSlot;
class WorkloadDataCollector;

class OutputHandler
{
public:
    /// @brief - Sets the TensorInfo used by this output handler.
    /// @param tensorInfo - TensorInfo for the output.
    void SetTensorInfo(const TensorInfo& tensorInfo);

    /// @brief - Creates tensor handles used by the intermediate tensors. Does not allocate memory.
    /// @param factory - Factory to be used for handler creation.
    void CreateTensorHandles(const IWorkloadFactory& factory, const bool IsMemoryManaged = true);
    void CreateTensorHandles(const ITensorHandleFactory& factory, const bool IsMemoryManaged = true);

    /// @brief - Gets the matching TensorInfo for the output.
    /// @return - References to the output TensorInfo.
    const TensorInfo& GetTensorInfo() const { return m_TensorInfo; }

    /// @brief - Gets the allocated tensor memory.
    /// @return - Pointer to the tensor memory.
    ITensorHandle* GetData() const { return m_TensorHandle.get(); }

    /// Fill the outputs for a given queue descriptor.
    void CollectWorkloadOutputs(WorkloadDataCollector& dataCollector) const;

    void SetData(std::unique_ptr<ITensorHandle> data) { m_TensorHandle = std::move(data); }

    void SetAllocatedData();

    void UseAllocatedData() { m_TensorHandle = m_AllocatedTensorHandle; }

    /// @brief Returns true if SetTensorInfo() has been called at least once on this.
    bool IsTensorInfoSet() const { return m_bTensorInfoSet; }
private:
    std::shared_ptr<ITensorHandle> m_TensorHandle;
    std::shared_ptr<ITensorHandle> m_AllocatedTensorHandle;
    TensorInfo m_TensorInfo;
    bool m_bTensorInfoSet = false;
};

} //namespace armnn
