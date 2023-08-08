//
// Copyright © 2021-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ITensorHandle.hpp"
#include "Workload.hpp"

#include <armnn/Optional.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/TensorFwd.hpp>

#include <memory>

namespace armnn
{

class Layer;

// Workload factory interface for compute backends.
class IWorkloadFactory
{
public:
    virtual ~IWorkloadFactory() { }

    virtual void AfterWorkloadsCreated() {};

    virtual const BackendId& GetBackendId() const = 0;

    static bool IsLayerSupported(const BackendId& backendId,
                                 const IConnectableLayer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    static bool IsLayerSupported(const IConnectableLayer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    static bool IsLayerSupported(const IConnectableLayer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported,
                                 const ModelOptions& modelOptions);

    static bool IsLayerSupported(const BackendId& backendId,
                                 const IConnectableLayer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported,
                                 const ModelOptions& modelOptions);

    virtual bool SupportsSubTensors() const = 0;

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateSubTensorHandle instead")
    virtual std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                                 TensorShape const& subTensorShape,
                                                                 unsigned int const* subTensorOrigin
                                                                ) const = 0;

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                              const bool IsMemoryManaged = true) const = 0;

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                              DataLayout dataLayout,
                                                              const bool IsMemoryManaged = true) const = 0;

    /// Backends should implement their own CreateWorkload function with a switch statement.
    /// The case for the switch should be the LayerType and based on that they will call their
    /// specific workload creation functionality.
    virtual std::unique_ptr<IWorkload> CreateWorkload(LayerType type,
                                                      const QueueDescriptor& descriptor,
                                                      const WorkloadInfo& info) const = 0;

private:
    static bool IsLayerConfigurationSupported(const BackendId& backendId,
                                              const IConnectableLayer& connectableLayer,
                                              Optional<DataType> dataType,
                                              std::string& outReasonIfUnsupported,
                                              const ModelOptions& modelOptions = {});
};

} // namespace armnn
