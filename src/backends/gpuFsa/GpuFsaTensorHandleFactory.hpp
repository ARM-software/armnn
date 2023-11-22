//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/ITensorHandleFactory.hpp>

#include <aclCommon/BaseMemoryManager.hpp>

namespace armnn
{

constexpr const char * GpuFsaTensorHandleFactoryId() { return "Arm/GpuFsa/TensorHandleFactory"; }

class GpuFsaTensorHandleFactory : public ITensorHandleFactory
{

public:
    GpuFsaTensorHandleFactory(std::shared_ptr<GpuFsaMemoryManager> mgr)
    : m_MemoryManager(mgr)
    {}

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override;

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
    mutable std::shared_ptr<GpuFsaMemoryManager> m_MemoryManager;

};

} // namespace armnn