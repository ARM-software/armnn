//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

#include "RefBaseWorkload.hpp"

namespace armnn
{

template <armnn::DataType DataType>
class RefDebugWorkload : public TypedWorkload<DebugQueueDescriptor, DataType>
{
public:
    RefDebugWorkload(const DebugQueueDescriptor& descriptor, const WorkloadInfo& info)
    : TypedWorkload<DebugQueueDescriptor, DataType>(descriptor, info)
    , m_Callback(nullptr) {}

    static const std::string& GetName()
    {
        static const std::string name = std::string("RefDebug") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    using TypedWorkload<DebugQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<DebugQueueDescriptor, DataType>::TypedWorkload;

    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

    void RegisterDebugCallback(const DebugCallbackFunction& func) override;

private:
    void Execute(std::vector<ITensorHandle*> inputs) const;
    DebugCallbackFunction m_Callback;
};

using RefDebugBFloat16Workload   = RefDebugWorkload<DataType::BFloat16>;
using RefDebugFloat16Workload   = RefDebugWorkload<DataType::Float16>;
using RefDebugFloat32Workload   = RefDebugWorkload<DataType::Float32>;
using RefDebugQAsymmU8Workload  = RefDebugWorkload<DataType::QAsymmU8>;
using RefDebugQAsymmS8Workload  = RefDebugWorkload<DataType::QAsymmS8>;
using RefDebugQSymmS16Workload  = RefDebugWorkload<DataType::QSymmS16>;
using RefDebugQSymmS8Workload   = RefDebugWorkload<DataType::QSymmS8>;
using RefDebugSigned32Workload  = RefDebugWorkload<DataType::Signed32>;

} // namespace armnn
