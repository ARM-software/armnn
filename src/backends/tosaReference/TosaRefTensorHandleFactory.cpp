//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaRefTensorHandleFactory.hpp"
#include "TosaRefTensorHandle.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

using FactoryId = ITensorHandleFactory::FactoryId;

const FactoryId& TosaRefTensorHandleFactory::GetIdStatic()
{
    static const FactoryId s_Id(TosaRefTensorHandleFactoryId());
    return s_Id;
}

std::unique_ptr<ITensorHandle> TosaRefTensorHandleFactory::CreateSubTensorHandle(ITensorHandle& parent,
                                                                                 const TensorShape& subTensorShape,
                                                                                 const unsigned int* subTensorOrigin)
                                                                                 const
{
    IgnoreUnused(parent, subTensorShape, subTensorOrigin);
    return nullptr;
}

std::unique_ptr<ITensorHandle> TosaRefTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return std::make_unique<TosaRefTensorHandle>(tensorInfo, m_MemoryManager);
}

std::unique_ptr<ITensorHandle> TosaRefTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                              DataLayout dataLayout) const
{
    IgnoreUnused(dataLayout);
    return std::make_unique<TosaRefTensorHandle>(tensorInfo, m_MemoryManager);
}

std::unique_ptr<ITensorHandle> TosaRefTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                              const bool IsMemoryManaged) const
{
    if (IsMemoryManaged)
    {
        return std::make_unique<TosaRefTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<TosaRefTensorHandle>(tensorInfo, m_ImportFlags);
    }
}

std::unique_ptr<ITensorHandle> TosaRefTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                              DataLayout dataLayout,
                                                                              const bool IsMemoryManaged) const
{
    IgnoreUnused(dataLayout);
    if (IsMemoryManaged)
    {
        return std::make_unique<TosaRefTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<TosaRefTensorHandle>(tensorInfo, m_ImportFlags);
    }
}

const FactoryId& TosaRefTensorHandleFactory::GetId() const
{
    return GetIdStatic();
}

bool TosaRefTensorHandleFactory::SupportsSubTensors() const
{
    return false;
}

MemorySourceFlags TosaRefTensorHandleFactory::GetExportFlags() const
{
    return m_ExportFlags;
}

MemorySourceFlags TosaRefTensorHandleFactory::GetImportFlags() const
{
    return m_ImportFlags;
}

} // namespace armnn