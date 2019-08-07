//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefTensorHandleFactory.hpp"
#include "RefTensorHandle.hpp"

#include <boost/core/ignore_unused.hpp>

namespace armnn
{

using FactoryId = ITensorHandleFactory::FactoryId;

const FactoryId& RefTensorHandleFactory::GetIdStatic()
{
    static const FactoryId s_Id(RefTensorHandleFactoryId());
    return s_Id;
}

std::unique_ptr<ITensorHandle> RefTensorHandleFactory::CreateSubTensorHandle(ITensorHandle& parent,
                                                                             TensorShape const& subTensorShape,
                                                                             unsigned int const* subTensorOrigin) const
{
    boost::ignore_unused(parent, subTensorShape, subTensorOrigin);
    return nullptr;
}

std::unique_ptr<ITensorHandle> RefTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return std::make_unique<RefTensorHandle>(tensorInfo, m_MemoryManager, m_ImportFlags);
}

std::unique_ptr<ITensorHandle> RefTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                          DataLayout dataLayout) const
{
    boost::ignore_unused(dataLayout);
    return std::make_unique<RefTensorHandle>(tensorInfo, m_MemoryManager, m_ImportFlags);
}

const FactoryId& RefTensorHandleFactory::GetId() const
{
    return GetIdStatic();
}

bool RefTensorHandleFactory::SupportsSubTensors() const
{
    return false;
}

MemorySourceFlags RefTensorHandleFactory::GetExportFlags() const
{
    return m_ExportFlags;
}

MemorySourceFlags RefTensorHandleFactory::GetImportFlags() const
{
    return m_ImportFlags;
}

} // namespace armnn