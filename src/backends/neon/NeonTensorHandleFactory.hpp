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

const std::set<armnn::LayerType> paddingRequiredLayers {
    LayerType::ArgMinMax,
    LayerType::Concat,
    LayerType::Convolution2d,
    LayerType::DepthToSpace,
    LayerType::DepthwiseConvolution2d,
    LayerType::Dequantize,
    LayerType::FullyConnected,
    LayerType::Gather,
    LayerType::L2Normalization,
    LayerType::Lstm,
    LayerType::Mean,
    LayerType::Multiplication,
    LayerType::Normalization,
    LayerType::Permute,
    LayerType::Pooling2d,
    LayerType::Quantize,
    LayerType::QuantizedLstm,
    LayerType::Resize,
    LayerType::Stack,
    LayerType::Transpose,
    LayerType::TransposeConvolution2d
};

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

    std::vector<Capability> GetCapabilities(const IConnectableLayer* layer,
                                            const IConnectableLayer* connectedLayer,
                                            CapabilityClass capabilityClass) override;

private:
    mutable std::shared_ptr<NeonMemoryManager> m_MemoryManager;
};

} // namespace armnn
