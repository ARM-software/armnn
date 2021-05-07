//
// Copyright © 2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <aclCommon/BaseMemoryManager.hpp>
#include <armnn/MemorySources.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/backends/ITensorHandleFactory.hpp>

namespace armnn
{

constexpr const char* ClImportTensorHandleFactoryId()
{
    return "Arm/Cl/ImportTensorHandleFactory";
}

/**
 * This factory creates ClImportTensorHandles that refer to imported memory tensors.
 */
class ClImportTensorHandleFactory : public ITensorHandleFactory
{
public:
    static const FactoryId m_Id;

    /**
     * Create a tensor handle factory for tensors that will be imported or exported.
     *
     * @param importFlags
     * @param exportFlags
     */
    ClImportTensorHandleFactory(MemorySourceFlags importFlags, MemorySourceFlags exportFlags)
        : m_ImportFlags(importFlags)
        , m_ExportFlags(exportFlags)
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

    bool SupportsMapUnmap() const override;

    MemorySourceFlags GetExportFlags() const override;

    MemorySourceFlags GetImportFlags() const override;

    std::vector<Capability> GetCapabilities(const IConnectableLayer* layer,
                                            const IConnectableLayer* connectedLayer,
                                            CapabilityClass capabilityClass) override;

private:
    MemorySourceFlags m_ImportFlags;
    MemorySourceFlags m_ExportFlags;
};

}    // namespace armnn