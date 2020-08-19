//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SampleDynamicTensorHandleFactory.hpp"
#include "SampleTensorHandle.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

namespace sdb // sample dynamic backend
{

using FactoryId = armnn::ITensorHandleFactory::FactoryId;

const FactoryId& SampleDynamicTensorHandleFactory::GetIdStatic()
{
    static const FactoryId s_Id(SampleDynamicTensorHandleFactoryId());
    return s_Id;
}

std::unique_ptr<armnn::ITensorHandle>
SampleDynamicTensorHandleFactory::CreateSubTensorHandle(armnn::ITensorHandle& parent,
                                                        armnn::TensorShape const& subTensorShape,
                                                        unsigned int const* subTensorOrigin) const
{
    IgnoreUnused(parent, subTensorShape, subTensorOrigin);
    return nullptr;
}

std::unique_ptr<armnn::ITensorHandle> SampleDynamicTensorHandleFactory::CreateTensorHandle(
        const armnn::TensorInfo& tensorInfo) const
{
    return std::make_unique<SampleTensorHandle>(tensorInfo, m_MemoryManager);
}

std::unique_ptr<armnn::ITensorHandle> SampleDynamicTensorHandleFactory::CreateTensorHandle(
        const armnn::TensorInfo& tensorInfo,
        armnn::DataLayout dataLayout) const
{
    IgnoreUnused(dataLayout);
    return std::make_unique<SampleTensorHandle>(tensorInfo, m_MemoryManager);
}

std::unique_ptr<armnn::ITensorHandle> SampleDynamicTensorHandleFactory::CreateTensorHandle(
        const armnn::TensorInfo& tensorInfo,
        const bool IsMemoryManaged) const
{
    if (IsMemoryManaged)
    {
        return std::make_unique<SampleTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<SampleTensorHandle>(tensorInfo, m_ImportFlags);
    }
}

std::unique_ptr<armnn::ITensorHandle> SampleDynamicTensorHandleFactory::CreateTensorHandle(
        const armnn::TensorInfo& tensorInfo,
        armnn::DataLayout dataLayout,
        const bool IsMemoryManaged) const
{
    IgnoreUnused(dataLayout);
    if (IsMemoryManaged)
    {
        return std::make_unique<SampleTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<SampleTensorHandle>(tensorInfo, m_ImportFlags);
    }
}

const FactoryId& SampleDynamicTensorHandleFactory::GetId() const
{
    return GetIdStatic();
}

bool SampleDynamicTensorHandleFactory::SupportsSubTensors() const
{
    return false;
}

armnn::MemorySourceFlags SampleDynamicTensorHandleFactory::GetExportFlags() const
{
    return m_ExportFlags;
}

armnn::MemorySourceFlags SampleDynamicTensorHandleFactory::GetImportFlags() const
{
    return m_ImportFlags;
}

} // namespace sdb