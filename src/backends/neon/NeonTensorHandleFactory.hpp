//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <aclCommon/BaseMemoryManager.hpp>
#include <backendsCommon/ITensorHandleFactory.hpp>

namespace armnn
{

class NeonTensorHandleFactory : public ITensorHandleFactory
{
public:
    NeonTensorHandleFactory(std::weak_ptr<NeonMemoryManager> mgr, ITensorHandleFactory::FactoryId id)
        : m_Id(id)
        , m_MemoryManager(mgr)
    {}

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout) const override;

    const FactoryId GetId() const override;

    bool SupportsSubTensors() const override;

    MemorySourceFlags GetExportFlags() const override;

    MemorySourceFlags GetImportFlags() const override;

private:
    FactoryId m_Id = "Arm/Neon/TensorHandleFactory";
    MemorySourceFlags m_ImportFlags;
    MemorySourceFlags m_ExportFlags;
    mutable std::shared_ptr<NeonMemoryManager> m_MemoryManager;
};

} // namespace armnn
