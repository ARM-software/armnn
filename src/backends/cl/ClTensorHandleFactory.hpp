//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <aclCommon/BaseMemoryManager.hpp>
#include <armnn/MemorySources.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/backends/ITensorHandleFactory.hpp>

namespace armnn
{

constexpr const char* ClTensorHandleFactoryId()
{
    return "Arm/Cl/TensorHandleFactory";
}

class ClTensorHandleFactory : public ITensorHandleFactory
{
public:
    static const FactoryId m_Id;

    ClTensorHandleFactory(std::shared_ptr<ClMemoryManager> mgr)
        : m_MemoryManager(mgr)
    {}

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         const TensorShape& subTensorShape,
                                                         const unsigned int* subTensorOrigin) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      const bool IsMemoryManaged) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
                                                      const bool IsMemoryManaged) const override;

    static const FactoryId& GetIdStatic();

    const FactoryId& GetId() const override;

    bool SupportsSubTensors() const override;

    MemorySourceFlags GetExportFlags() const override;

    MemorySourceFlags GetImportFlags() const override;

private:
    mutable std::shared_ptr<ClMemoryManager> m_MemoryManager;
};

}    // namespace armnn
