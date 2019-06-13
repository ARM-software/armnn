//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <armnn/IRuntime.hpp>

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

    virtual const FactoryId GetId() const = 0;

    virtual bool SupportsSubTensors() const = 0;

    virtual bool SupportsMapUnmap() const final { return true; }

    virtual bool SupportsExport() const final { return false; }

    virtual bool SupportsImport() const final { return false; }
};

enum class MemoryStrategy
{
    Undefined,
    DirectCompatibility,    // Only allocate the tensorhandle using the assigned factory
    CopyToTarget,           // Default + Insert MemCopy node before target
    ExportToTarget,         // Default + Insert Import node
};

} //namespace armnn
