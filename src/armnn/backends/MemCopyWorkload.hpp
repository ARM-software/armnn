//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "CpuTensorHandleFwd.hpp"
#include "backends/Workload.hpp"

#include <utility>

namespace armnn
{

template <armnn::DataType DataType>
class CopyFromCpuToCpuWorkload : public TypedWorkload<MemCopyQueueDescriptor, DataType>
{
public:
    CopyFromCpuToCpuWorkload(const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TensorHandlePair = std::pair<const ConstCpuTensorHandle*, CpuTensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

using CopyFromCpuToCpuFloat32Workload = CopyFromCpuToCpuWorkload<DataType::Float32>;
using CopyFromCpuToCpuUint8Workload = CopyFromCpuToCpuWorkload<DataType::QuantisedAsymm8>;

#if ARMCOMPUTECL_ENABLED

class IClTensorHandle;

template <armnn::DataType DataType>
class CopyFromCpuToClWorkload : public TypedWorkload<MemCopyQueueDescriptor, DataType>
{
public:
    CopyFromCpuToClWorkload(const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TensorHandlePair = std::pair<const ConstCpuTensorHandle*, IClTensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

using CopyFromCpuToClFloat32Workload = CopyFromCpuToClWorkload<DataType::Float32>;
using CopyFromCpuToClUint8Workload = CopyFromCpuToClWorkload<DataType::QuantisedAsymm8>;

template <armnn::DataType DataType>
class CopyFromClToCpuWorkload : public TypedWorkload<MemCopyQueueDescriptor, DataType>
{
public:
    CopyFromClToCpuWorkload(const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TensorHandlePair = std::pair<IClTensorHandle*, CpuTensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

using CopyFromClToCpuFloat32Workload = CopyFromClToCpuWorkload<DataType::Float32>;
using CopyFromClToCpuUint8Workload = CopyFromClToCpuWorkload<DataType::QuantisedAsymm8>;

#endif // ARMCOMPUTECL_ENABLED

#if ARMCOMPUTENEON_ENABLED

class INeonTensorHandle;

template <armnn::DataType DataType>
class CopyFromCpuToNeonWorkload : public TypedWorkload<MemCopyQueueDescriptor, DataType>
{
public:
    CopyFromCpuToNeonWorkload(const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

protected:
    using TensorHandlePair = std::pair<const ConstCpuTensorHandle*, INeonTensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

using CopyFromCpuToNeonFloat32Workload = CopyFromCpuToNeonWorkload<DataType::Float32>;
using CopyFromCpuToNeonUint8Workload = CopyFromCpuToNeonWorkload<DataType::QuantisedAsymm8>;

template <armnn::DataType DataType>
class CopyFromNeonToCpuWorkload : public TypedWorkload<MemCopyQueueDescriptor, DataType>
{
public:
    CopyFromNeonToCpuWorkload(const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

protected:
    using TensorHandlePair = std::pair<const INeonTensorHandle*, CpuTensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

using CopyFromNeonToCpuFloat32Workload = CopyFromNeonToCpuWorkload<DataType::Float32>;
using CopyFromNeonToCpuUint8Workload = CopyFromNeonToCpuWorkload<DataType::QuantisedAsymm8>;

#endif

#if ARMCOMPUTECL_ENABLED && ARMCOMPUTENEON_ENABLED

template <armnn::DataType DataType>
class CopyFromNeonToClWorkload : public TypedWorkload<MemCopyQueueDescriptor, DataType>
{
public:
    CopyFromNeonToClWorkload(const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TensorHandlePair = std::pair<const INeonTensorHandle*, IClTensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

using CopyFromNeonToClFloat32Workload = CopyFromNeonToClWorkload<DataType::Float32>;
using CopyFromNeonToClUint8Workload = CopyFromNeonToClWorkload<DataType::QuantisedAsymm8>;

template <armnn::DataType DataType>
class CopyFromClToNeonWorkload : public TypedWorkload<MemCopyQueueDescriptor, DataType>
{
public:
    CopyFromClToNeonWorkload(const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TensorHandlePair = std::pair<IClTensorHandle*, INeonTensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

using CopyFromClToNeonFloat32Workload = CopyFromClToNeonWorkload<DataType::Float32>;
using CopyFromClToNeonUint8Workload = CopyFromClToNeonWorkload<DataType::QuantisedAsymm8>;

#endif

}
