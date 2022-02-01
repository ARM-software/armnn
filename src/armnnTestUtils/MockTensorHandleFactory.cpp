//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MockTensorHandleFactory.hpp"
#include <armnnTestUtils/MockTensorHandle.hpp>

namespace armnn
{

using FactoryId = ITensorHandleFactory::FactoryId;

const FactoryId& MockTensorHandleFactory::GetIdStatic()
{
    static const FactoryId s_Id(MockTensorHandleFactoryId());
    return s_Id;
}

std::unique_ptr<ITensorHandle> MockTensorHandleFactory::CreateSubTensorHandle(ITensorHandle&,
                                                                              TensorShape const&,
                                                                              unsigned int const*) const
{
    return nullptr;
}

std::unique_ptr<ITensorHandle> MockTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return std::make_unique<MockTensorHandle>(tensorInfo, m_MemoryManager);
}

std::unique_ptr<ITensorHandle> MockTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           DataLayout dataLayout) const
{
    IgnoreUnused(dataLayout);
    return std::make_unique<MockTensorHandle>(tensorInfo, m_MemoryManager);
}

std::unique_ptr<ITensorHandle> MockTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           const bool IsMemoryManaged) const
{
    if (IsMemoryManaged)
    {
        return std::make_unique<MockTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<MockTensorHandle>(tensorInfo, m_ImportFlags);
    }
}

std::unique_ptr<ITensorHandle> MockTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           DataLayout dataLayout,
                                                                           const bool IsMemoryManaged) const
{
    IgnoreUnused(dataLayout);
    if (IsMemoryManaged)
    {
        return std::make_unique<MockTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<MockTensorHandle>(tensorInfo, m_ImportFlags);
    }
}

const FactoryId& MockTensorHandleFactory::GetId() const
{
    return GetIdStatic();
}

bool MockTensorHandleFactory::SupportsSubTensors() const
{
    return false;
}

MemorySourceFlags MockTensorHandleFactory::GetExportFlags() const
{
    return m_ExportFlags;
}

MemorySourceFlags MockTensorHandleFactory::GetImportFlags() const
{
    return m_ImportFlags;
}

}    // namespace armnn