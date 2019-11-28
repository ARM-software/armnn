//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <aclCommon/BaseMemoryManager.hpp>
#include <armnn/backends/ITensorHandleFactory.hpp>

namespace armnn
{

constexpr const char* NeonTensorHandleFactoryId() { return "Arm/Neon/TensorHandleFactory"; }

class NeonTensorHandleFactory : public ITensorHandleFactory
{
public:
    NeonTensorHandleFactory(std::weak_ptr<NeonMemoryManager> mgr)
                            : m_MemoryManager(mgr)
    {}

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         const TensorShape& subTensorShape,
                                                         const unsigned int* subTensorOrigin) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      const bool IsMemoryManaged = true) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
                                                      const bool IsMemoryManaged = true) const override;

    static const FactoryId& GetIdStatic();

    const FactoryId& GetId() const override;

    bool SupportsSubTensors() const override;

    MemorySourceFlags GetExportFlags() const override;

    MemorySourceFlags GetImportFlags() const override;

private:
    mutable std::shared_ptr<NeonMemoryManager> m_MemoryManager;
};

} // namespace armnn
