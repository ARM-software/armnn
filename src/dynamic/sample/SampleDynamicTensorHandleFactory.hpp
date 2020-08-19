//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "SampleMemoryManager.hpp"

#include <armnn/backends/ITensorHandleFactory.hpp>

namespace sdb // sample dynamic backend
{

constexpr const char * SampleDynamicTensorHandleFactoryId() { return "Arm/SampleDynamic/TensorHandleFactory"; }

class SampleDynamicTensorHandleFactory : public armnn::ITensorHandleFactory
{

public:
    SampleDynamicTensorHandleFactory(std::shared_ptr<SampleMemoryManager> mgr)
    : m_MemoryManager(mgr),
      m_ImportFlags(static_cast<armnn::MemorySourceFlags>(armnn::MemorySource::Malloc)),
      m_ExportFlags(static_cast<armnn::MemorySourceFlags>(armnn::MemorySource::Malloc))
    {}

    std::unique_ptr<armnn::ITensorHandle> CreateSubTensorHandle(armnn::ITensorHandle& parent,
                                                                armnn::TensorShape const& subTensorShape,
                                                                unsigned int const* subTensorOrigin) const override;

    std::unique_ptr<armnn::ITensorHandle> CreateTensorHandle(const armnn::TensorInfo& tensorInfo) const override;

    std::unique_ptr<armnn::ITensorHandle> CreateTensorHandle(const armnn::TensorInfo& tensorInfo,
                                                             armnn::DataLayout dataLayout) const override;

    std::unique_ptr<armnn::ITensorHandle> CreateTensorHandle(const armnn::TensorInfo& tensorInfo,
                                                             const bool IsMemoryManaged) const override;

    std::unique_ptr<armnn::ITensorHandle> CreateTensorHandle(const armnn::TensorInfo& tensorInfo,
                                                             armnn::DataLayout dataLayout,
                                                             const bool IsMemoryManaged) const override;

    static const FactoryId& GetIdStatic();

    const FactoryId& GetId() const override;

    bool SupportsSubTensors() const override;

    armnn::MemorySourceFlags GetExportFlags() const override;

    armnn::MemorySourceFlags GetImportFlags() const override;

private:
    mutable std::shared_ptr<SampleMemoryManager> m_MemoryManager;
    armnn::MemorySourceFlags m_ImportFlags;
    armnn::MemorySourceFlags m_ExportFlags;
};

} // namespace sdb

