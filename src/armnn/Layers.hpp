//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayersFwd.hpp"

#include "Layer.hpp"
#include "InternalTypes.hpp"

#include <armnn/Descriptors.hpp>

#include <boost/core/ignore_unused.hpp>

namespace armnn
{

class ScopedCpuTensorHandle;

template <typename Parameters>
class LayerWithParameters : public Layer
{
public:
    using DescriptorType = Parameters;

    const Parameters& GetParameters() const { return m_Param; }

    /// Helper to serialize the layer parameters to string
    /// (currently used in DotSerializer and company)
    void SerializeLayerParameters(ParameterStringifyFunction & fn) const
    {
        StringifyLayerParameters<Parameters>::Serialize(fn, m_Param);
    }

protected:
    LayerWithParameters(unsigned int numInputSlots,
                        unsigned int numOutputSlots,
                        LayerType type,
                        const Parameters& param,
                        const char* name)
    :   Layer(numInputSlots, numOutputSlots, type, name)
    ,   m_Param(param)
    {
    }

    ~LayerWithParameters() = default;

    /// Helper function to reduce duplication in *Layer::CreateWorkload
    template <typename QueueDescriptor>
    WorkloadInfo PrepInfoAndDesc(QueueDescriptor& descriptor, const Graph& graph) const
    {
        descriptor.m_Parameters = m_Param;
        return Layer::PrepInfoAndDesc(descriptor, graph);
    }

    /// The parameters for the layer (not including tensor-valued weights etc.)
    Parameters m_Param;
};

class ActivationLayer : public LayerWithParameters<ActivationDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    ActivationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    ActivationLayer(const ActivationDescriptor &param, const char* name);
    ~ActivationLayer() = default;
};

class AdditionLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    AdditionLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    AdditionLayer(const char* name);
    ~AdditionLayer() = default;
};

class BatchNormalizationLayer : public LayerWithParameters<BatchNormalizationDescriptor>
{
public:
    std::unique_ptr<ScopedCpuTensorHandle> m_Mean;
    std::unique_ptr<ScopedCpuTensorHandle> m_Variance;
    std::unique_ptr<ScopedCpuTensorHandle> m_Beta;
    std::unique_ptr<ScopedCpuTensorHandle> m_Gamma;

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    BatchNormalizationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    BatchNormalizationLayer(const BatchNormalizationDescriptor& param, const char* name);
    ~BatchNormalizationLayer() = default;
};

class Convolution2dLayer : public LayerWithParameters<Convolution2dDescriptor>
{
public:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    Convolution2dLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    Convolution2dLayer(const Convolution2dDescriptor& param, const char* name);
    ~Convolution2dLayer() = default;
};

class DepthwiseConvolution2dLayer : public LayerWithParameters<DepthwiseConvolution2dDescriptor>
{
public:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    DepthwiseConvolution2dLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    DepthwiseConvolution2dLayer(const DepthwiseConvolution2dDescriptor& param, const char* name);
    ~DepthwiseConvolution2dLayer() = default;
};

class FakeQuantizationLayer : public LayerWithParameters<FakeQuantizationDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    FakeQuantizationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    FakeQuantizationLayer(const FakeQuantizationDescriptor& descriptor, const char* name);
    ~FakeQuantizationLayer() = default;
};

class FloorLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    FloorLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    FloorLayer(const char* name);
    ~FloorLayer() = default;
};

class FullyConnectedLayer : public LayerWithParameters<FullyConnectedDescriptor>
{
public:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    FullyConnectedLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    FullyConnectedLayer(const FullyConnectedDescriptor& param, const char* name);
    ~FullyConnectedLayer() = default;
};

class InputLayer : public BindableLayer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    InputLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    InputLayer(LayerBindingId id, const char* name);
    ~InputLayer() = default;
};

class MergerLayer : public LayerWithParameters<OriginsDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;
    virtual void CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory) override;

    MergerLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    MergerLayer(const OriginsDescriptor& param, const char* name);
    ~MergerLayer() = default;
};

class MultiplicationLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    MultiplicationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    MultiplicationLayer(const char* name);
    ~MultiplicationLayer() = default;
};

class NormalizationLayer : public LayerWithParameters<NormalizationDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    NormalizationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    NormalizationLayer(const NormalizationDescriptor& param, const char* name);
    ~NormalizationLayer() = default;
};

class OutputLayer : public BindableLayer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;
    virtual void CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory) override
    {
        boost::ignore_unused(graph, factory);
    }

    OutputLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    OutputLayer(LayerBindingId id, const char* name);
    ~OutputLayer() = default;
};

class PermuteLayer : public LayerWithParameters<PermuteDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    PermuteLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

    const PermutationVector& GetPermutation() const
    {
        return m_Param.m_DimMappings;
    }

    bool IsInverse(const Layer& other) const
    {
        return (other.GetType() == LayerType::Permute) &&
            GetPermutation().IsInverse(boost::polymorphic_downcast<const PermuteLayer*>(&other)->GetPermutation());
    }

    bool IsEqual(const Layer& other) const
    {
        return (other.GetType() == LayerType::Permute) &&
               GetPermutation().IsEqual(boost::polymorphic_downcast<const PermuteLayer*>(&other)->GetPermutation());
    }

protected:
    PermuteLayer(const PermuteDescriptor& param, const char* name);
    ~PermuteLayer() = default;
};

class Pooling2dLayer : public LayerWithParameters<Pooling2dDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    Pooling2dLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    Pooling2dLayer(const Pooling2dDescriptor& param, const char* name);
    ~Pooling2dLayer() = default;
};

class SoftmaxLayer : public LayerWithParameters<SoftmaxDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    SoftmaxLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    SoftmaxLayer(const SoftmaxDescriptor& param, const char* name);
    ~SoftmaxLayer() = default;
};

class SplitterLayer : public LayerWithParameters<ViewsDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;
    virtual void CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory) override;

    SplitterLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    SplitterLayer(const ViewsDescriptor& param, const char* name);
    ~SplitterLayer() = default;
};

class MemCopyLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload>
    CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const override;

    MemCopyLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    MemCopyLayer(const char* name);
    ~MemCopyLayer() = default;
};

class ResizeBilinearLayer : public LayerWithParameters<ResizeBilinearDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload>
        CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const override;

    ResizeBilinearLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    ResizeBilinearLayer(const ResizeBilinearDescriptor& param, const char* name);
    ~ResizeBilinearLayer() = default;
};

class L2NormalizationLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
        const IWorkloadFactory& factory) const override;

    L2NormalizationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    L2NormalizationLayer(const char* name);
    ~L2NormalizationLayer() = default;
};

class ConstantLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
        const IWorkloadFactory& factory) const override;

    ConstantLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    ConstantLayer(const std::shared_ptr<ScopedCpuTensorHandle>& input, const char* name);
    ~ConstantLayer() = default;

private:
    std::shared_ptr<ScopedCpuTensorHandle> m_LayerOutput;
};

class ReshapeLayer : public LayerWithParameters<ReshapeDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
        const IWorkloadFactory& factory) const override;

    ReshapeLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

    bool IsEqual(const Layer& other) const
    {
        return (other.GetType() == LayerType::Reshape) &&
               m_Param.m_TargetShape == boost::polymorphic_downcast<const ReshapeLayer*>(&other)->m_Param.m_TargetShape;
    }

protected:
    ReshapeLayer(const ReshapeDescriptor& desc, const char* name);
    ~ReshapeLayer() = default;
};

}
