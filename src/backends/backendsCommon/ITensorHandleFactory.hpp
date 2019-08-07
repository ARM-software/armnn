//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/IRuntime.hpp>
#include <armnn/MemorySources.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

class ITensorHandleFactory
{
public:
    using FactoryId = std::string;
    static const FactoryId LegacyFactoryId;   // Use the workload factory to create the tensor handle
    static const FactoryId DeferredFactoryId; // Some TensorHandleFactory decisions are deferred to run-time

    virtual ~ITensorHandleFactory() {}

    virtual std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                                 TensorShape const& subTensorShape,
                                                                 unsigned int const* subTensorOrigin) const = 0;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const = 0;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                              DataLayout dataLayout) const = 0;

    virtual const FactoryId& GetId() const = 0;

    virtual bool SupportsSubTensors() const = 0;

    virtual bool SupportsMapUnmap() const final { return true; }

    virtual MemorySourceFlags GetExportFlags() const { return 0; }
    virtual MemorySourceFlags GetImportFlags() const { return 0; }
};

enum class EdgeStrategy
{
    Undefined,              /// No strategy has been defined. Used internally to verify integrity of optimizations.
    DirectCompatibility,    /// Destination backend can work directly with tensors on source backend.
    ExportToTarget,         /// Source backends tensor data can be exported to destination backend tensor without copy.
    CopyToTarget            /// Copy contents from source backend tensor to destination backend tensor.
};

} //namespace armnn
