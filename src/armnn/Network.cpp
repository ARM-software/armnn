//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Network.hpp"
#include "Graph.hpp"
#include "Layer.hpp"
#include "DeviceSpec.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/WorkloadFactory.hpp"
#include "Optimizer.hpp"
#include "armnn/Exceptions.hpp"

#include <armnn/Utils.hpp>
#include <armnn/TypesUtils.hpp>

#include <fcntl.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>

#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/converter_policies.hpp>
#include <boost/cast.hpp>

#include "optimizations/All.hpp"

namespace armnn
{

armnn::INetwork* INetwork::CreateRaw()
{
    return new Network();
}

armnn::INetworkPtr INetwork::Create()
{
    return INetworkPtr(CreateRaw(), &INetwork::Destroy);
}

void INetwork::Destroy(INetwork* network)
{
    delete boost::polymorphic_downcast<Network*>(network);
}

Status Network::PrintGraph()
{
    m_Graph->Print();
    return Status::Success;
}

void IOptimizedNetwork::Destroy(IOptimizedNetwork* network)
{
    delete boost::polymorphic_downcast<OptimizedNetwork*>(network);
}

Status OptimizedNetwork::PrintGraph()
{
    m_Graph->Print();
    return Status::Success;
}

Status OptimizedNetwork::SerializeToDot(std::ostream& stream) const
{
    return m_Graph->SerializeToDot(stream);
}

IOptimizedNetworkPtr Optimize(const INetwork& inNetwork,
                              const std::vector<armnn::Compute>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptions& options)
{
    if (backendPreferences.empty()) {
        throw armnn::InvalidArgumentException("Invoked Optimize with no backends specified");
    }
    const Network& network = *boost::polymorphic_downcast<const Network*>(&inNetwork);
    std::unique_ptr<Graph> graph = std::make_unique<Graph>(network.GetGraph());

    auto optNet = IOptimizedNetworkPtr(new OptimizedNetwork(std::move(graph)), &IOptimizedNetwork::Destroy);

    OptimizedNetwork* optNetObjPtr = boost::polymorphic_downcast<OptimizedNetwork*>(optNet.get());

    // Perform optimisation passes
    using namespace optimizations;
    Optimizer::Pass(optNetObjPtr->GetGraph(), MakeOptimizations(SquashEqualPermuteSiblings(),
                                                                SquashEqualReshapeSiblings(),
                                                                OptimizeInversePermutes(),
                                                                MovePermuteUp(),
                                                                PermuteAsReshape(),
                                                                OptimizeConsecutiveReshapes()));

    // Infer the tensor infos for all output slots. Throws an exception on failure.
    optNetObjPtr->GetGraph().InferTensorInfos();

    // if Fp32 to Fp16 optimization is set convert Fp32 network to Fp16
    if (options.m_ReduceFp32ToFp16)
    {
        Optimizer::Pass(optNetObjPtr->GetGraph(), MakeOptimizations(Fp32NetworkToFp16Converter()));
    }

    // We know that DeviceSpec should be the only implementation of IDeviceSpec.
    const DeviceSpec& spec = *boost::polymorphic_downcast<const DeviceSpec*>(&deviceSpec);

    // determine which of the preferred backends we have available for use
    // and whether we have specified CpuRef as one of those backends.
    bool cpuRefUsed = false;
    std::vector<armnn::Compute> availablePreferredBackends;
    for (const armnn::Compute& backend : backendPreferences)
    {
        // Check if the backend is in the available backend devices.
        if (std::find(spec.m_SupportedComputeDevices.begin(),
                      spec.m_SupportedComputeDevices.end(), backend) !=
                      spec.m_SupportedComputeDevices.end())
        {
            availablePreferredBackends.push_back(backend);
            if (armnn::Compute::CpuRef == backend) {
                cpuRefUsed = true;
            }
        }
    }
    if (availablePreferredBackends.empty()) {
        BOOST_LOG_TRIVIAL(warning) << "None of the preferred backends " << backendPreferences
                                   << " are supported. Current platform provides " << spec.m_SupportedComputeDevices;
        return {nullptr, &IOptimizedNetwork::Destroy};
    }

    auto ReturnWithError = [&](Layer* layer)
    {
        BOOST_LOG_TRIVIAL(warning) << "Layer of type " << GetLayerTypeAsCString(layer->GetType())
                    << " is not supported on any preferred backend " << backendPreferences;
        return IOptimizedNetworkPtr(nullptr, &IOptimizedNetwork::Destroy);
    };

    // Assign a compute device for all nodes
    for (auto&& layer : optNetObjPtr->GetGraph())
    {
        DataType dataType = layer->GetDataType();
        std::string reasonIfUnsupported;
        bool found = false;
        for (const armnn::Compute& backend : availablePreferredBackends)
        {
            // need to set the compute device on the layer
            // before we can check if it is supported
            layer->SetComputeDevice(backend);
            if (!IWorkloadFactory::IsLayerSupported(*layer, dataType, reasonIfUnsupported))
            {
                if (dataType == DataType::Float16)
                {
                    if (IWorkloadFactory::IsLayerSupported(*layer, DataType::Float32, reasonIfUnsupported)
                        && layer->GetType() != LayerType::ConvertFp32ToFp16
                        && layer->GetType() != LayerType::ConvertFp16ToFp32)
                    {
                        // Insert FP16 -> FP32 conversion layer before current layer
                        std::vector<ConvertFp16ToFp32Layer*> convertFp16ToFp32Layers =
                            InsertConvertFp16ToFp32LayersBefore(optNetObjPtr->GetGraph(), *layer);

                        // Insert FP32 -> FP16 conversion layer after current layer
                        std::vector<ConvertFp32ToFp16Layer*> convertFp32ToFp16Layers =
                            InsertConvertFp32ToFp16LayersAfter(optNetObjPtr->GetGraph(), *layer);

                        // Assign a supported backend to the newly introduced conversion layers
                        auto AssignFirstSupportedBackend = [&](Layer* layer, Compute preferredBackend)
                        {
                            bool supportedBackendFound = false;
                            std::string reasonIfUnsupported;

                            // Try preferred backend first
                            layer->SetComputeDevice(preferredBackend);
                            if (IWorkloadFactory::IsLayerSupported(*layer, boost::none, reasonIfUnsupported))
                            {
                                supportedBackendFound = true;
                            }
                            else
                            {
                                for (const Compute& backend : availablePreferredBackends)
                                {
                                    // Skip preferred backend (we already determined that it is not supported)
                                    if (backend == preferredBackend)
                                    {
                                        continue;
                                    }

                                    layer->SetComputeDevice(backend);
                                    if (IWorkloadFactory::IsLayerSupported(*layer, boost::none, reasonIfUnsupported))
                                    {
                                        supportedBackendFound = true;
                                        break;
                                    }
                                }
                            }

                            return supportedBackendFound;
                        };

                        for (ConvertFp16ToFp32Layer* convertLayer : convertFp16ToFp32Layers)
                        {
                            if (!AssignFirstSupportedBackend(convertLayer, backend))
                            {
                                return ReturnWithError(convertLayer);
                            }
                        }

                        for (ConvertFp32ToFp16Layer* convertLayer : convertFp32ToFp16Layers)
                        {
                            if (!AssignFirstSupportedBackend(convertLayer, backend))
                            {
                                return ReturnWithError(convertLayer);
                            }
                        }

                        found = true;
                        break;
                    }
                }
                BOOST_LOG_TRIVIAL(warning) << "Layer of type " << GetLayerTypeAsCString(layer->GetType())
                                           << " is not supported on requested backend " << layer->GetComputeDevice()
                                           << " (reason: " << reasonIfUnsupported
                                           << "), falling back to the next backend.";
            }
            else
            {
                found = true;
                break;
            }
        }

        // If the layer is unsupported by any devices, log and return a null network.
        if (!found) {
            // NOTE: if the layer is not an operation queue type AND we have not got CpuRef as a
            //       fallback we should set the compute device on the layer to CpuRef (these are not
            //       available as accelerated operations, or are only available under certain
            //       conditions, currently they comprise MemCopy, Constant, Permute)
            armnn::LayerType layerType = layer->GetType();
            if (!cpuRefUsed && (layerType == armnn::LayerType::MemCopy ||
                                layerType == armnn::LayerType::Constant ||
                                layerType == armnn::LayerType::Permute))
            {
                layer->SetComputeDevice(armnn::Compute::CpuRef);
            }
            else
            {
                return ReturnWithError(layer);
            }
        }
    }

    Optimizer::Pass(optNetObjPtr->GetGraph(), MakeOptimizations(OptimizeInverseConversionsFp16(),
                                                                OptimizeInverseConversionsFp32()));

    optNetObjPtr->GetGraph().AddCopyLayers();

    // Convert constants
    Optimizer::Pass(optNetObjPtr->GetGraph(), MakeOptimizations(ConvertConstantsFloatToHalf()));
    Optimizer::Pass(optNetObjPtr->GetGraph(), MakeOptimizations(ConvertConstantsHalfToFloat()));

    return optNet;
}

Network::Network()
: m_Graph(std::make_unique<Graph>())
{
}

Network::~Network()
{
}

IConnectableLayer* Network::AddInputLayer(LayerBindingId id, const char* name)
{
    return m_Graph->AddLayer<InputLayer>(id, name);
}

IConnectableLayer* Network::AddFullyConnectedLayerImpl(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                       const ConstTensor& weights,
                                                       const ConstTensor* biases,
                                                       const char* name)
{
    if (fullyConnectedDescriptor.m_BiasEnabled && (biases == nullptr))
    {
        throw InvalidArgumentException("AddFullyConnectedLayer: biases cannot be NULL");
    }

    const auto layer = m_Graph->AddLayer<FullyConnectedLayer>(fullyConnectedDescriptor, name);

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    if (fullyConnectedDescriptor.m_BiasEnabled)
    {
        layer->m_Bias = std::make_unique<ScopedCpuTensorHandle>(*biases);
    }

    return layer;
}

IConnectableLayer* Network::AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                   const ConstTensor& weights,
                                                   const char* name)
{
    return AddFullyConnectedLayerImpl(fullyConnectedDescriptor, weights, nullptr, name);
}

IConnectableLayer* Network::AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                   const ConstTensor& weights,
                                                   const ConstTensor& biases,
                                                   const char* name)
{
    return AddFullyConnectedLayerImpl(fullyConnectedDescriptor, weights, &biases, name);
}

IConnectableLayer* Network::AddConvolution2dLayerImpl(const Convolution2dDescriptor& convolution2dDescriptor,
                                                      const ConstTensor& weights,
                                                      const ConstTensor* biases,
                                                      const char* name)
{
    if (convolution2dDescriptor.m_BiasEnabled && (biases == nullptr))
    {
        throw InvalidArgumentException("AddConvolution2dLayer: biases cannot be NULL");
    }

    const auto layer = m_Graph->AddLayer<Convolution2dLayer>(convolution2dDescriptor, name);

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    if (convolution2dDescriptor.m_BiasEnabled)
    {
        layer->m_Bias = std::make_unique<ScopedCpuTensorHandle>(*biases);
    }

    return layer;
}

IConnectableLayer* Network::AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const char* name)
{
    return AddConvolution2dLayerImpl(convolution2dDescriptor, weights, nullptr, name);
}
IConnectableLayer* Network::AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const ConstTensor& biases,
                                                  const char* name)
{
    return AddConvolution2dLayerImpl(convolution2dDescriptor, weights, &biases, name);
}

IConnectableLayer* Network::AddDepthwiseConvolution2dLayerImpl(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const ConstTensor& weights,
    const ConstTensor* biases,
    const char* name)
{
    if (convolution2dDescriptor.m_BiasEnabled && (biases == nullptr))
    {
        throw InvalidArgumentException("AddDepthwiseConvolution2dLayer: biases cannot be NULL");
    }

    const auto layer = m_Graph->AddLayer<DepthwiseConvolution2dLayer>(convolution2dDescriptor,
            name);

    layer->m_Weight = std::make_unique<ScopedCpuTensorHandle>(weights);

    if (convolution2dDescriptor.m_BiasEnabled)
    {
        layer->m_Bias = std::make_unique<ScopedCpuTensorHandle>(*biases);
    }

    return layer;
}

IConnectableLayer* Network::AddDepthwiseConvolution2dLayer(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const ConstTensor& weights,
    const char* name)
{
    return AddDepthwiseConvolution2dLayerImpl(convolution2dDescriptor, weights, nullptr, name);
}
IConnectableLayer* Network::AddDepthwiseConvolution2dLayer(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const ConstTensor& weights,
    const ConstTensor& biases,
    const char* name)
{
    return AddDepthwiseConvolution2dLayerImpl(convolution2dDescriptor, weights, &biases, name);
}

IConnectableLayer* Network::AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                            const char* name)
{
    return m_Graph->AddLayer<PermuteLayer>(permuteDescriptor, name);
}

IConnectableLayer* Network::AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<Pooling2dLayer>(pooling2dDescriptor, name);
}

IConnectableLayer* Network::AddActivationLayer(const ActivationDescriptor& activationDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<ActivationLayer>(activationDescriptor, name);
}

IConnectableLayer* Network::AddNormalizationLayer(const NormalizationDescriptor&
normalizationDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<NormalizationLayer>(normalizationDescriptor, name);
}

IConnectableLayer* Network::AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<SoftmaxLayer>(softmaxDescriptor, name);
}

IConnectableLayer* Network::AddSplitterLayer(const ViewsDescriptor& splitterDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<SplitterLayer>(splitterDescriptor, name);
}

IConnectableLayer* Network::AddMergerLayer(const OriginsDescriptor& mergerDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<MergerLayer>(mergerDescriptor, name);
}

IConnectableLayer* Network::AddAdditionLayer(const char* name)
{
    return m_Graph->AddLayer<AdditionLayer>(name);
}

IConnectableLayer* Network::AddMultiplicationLayer(const char* name)
{
    return m_Graph->AddLayer<MultiplicationLayer>(name);
}

IConnectableLayer* Network::AddOutputLayer(LayerBindingId id, const char* name)
{
    return m_Graph->AddLayer<OutputLayer>(id, name);
}

IConnectableLayer* Network::AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
                                                       const ConstTensor&                  mean,
                                                       const ConstTensor&                  variance,
                                                       const ConstTensor&                  beta,
                                                       const ConstTensor&                  gamma,
                                                       const char*                         name)
{
    const auto layer = m_Graph->AddLayer<BatchNormalizationLayer>(desc, name);

    layer->m_Mean = std::make_unique<ScopedCpuTensorHandle>(mean);
    layer->m_Variance = std::make_unique<ScopedCpuTensorHandle>(variance);
    layer->m_Beta = std::make_unique<ScopedCpuTensorHandle>(beta);
    layer->m_Gamma = std::make_unique<ScopedCpuTensorHandle>(gamma);

    return layer;
}

IConnectableLayer* Network::AddResizeBilinearLayer(const ResizeBilinearDescriptor&
resizeDescriptor, const char* name)
{
    return m_Graph->AddLayer<ResizeBilinearLayer>(resizeDescriptor,name);
}

IConnectableLayer* Network::AddL2NormalizationLayer(const char* name)
{
    return m_Graph->AddLayer<L2NormalizationLayer>(name);
}

IConnectableLayer* Network::AddConstantLayer(const ConstTensor& input, const char* name)
{
    auto layer = m_Graph->AddLayer<ConstantLayer>(name);

    layer->m_LayerOutput = std::make_unique<ScopedCpuTensorHandle>(input);

    return layer;
}

IConnectableLayer* Network::AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                            const char* name)
{
    return m_Graph->AddLayer<ReshapeLayer>(reshapeDescriptor, name);
}

IConnectableLayer* Network::AddFloorLayer(const char* name)
{
    return m_Graph->AddLayer<FloorLayer>(name);
}

IConnectableLayer* Network::AddLstmLayer(const LstmDescriptor&  descriptor,
                                         const LstmInputParams& params,
                                         const char* name)
{
    const auto layer = m_Graph->AddLayer<LstmLayer>(descriptor, name);

    //Lstm Basic Parameters
    layer->m_BasicParameters.m_InputToForgetWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToForgetWeights));
    layer->m_BasicParameters.m_InputToCellWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToCellWeights));
    layer->m_BasicParameters.m_InputToOutputWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToOutputWeights));
    layer->m_BasicParameters.m_RecurrentToForgetWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToForgetWeights));
    layer->m_BasicParameters.m_RecurrentToCellWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToCellWeights));
    layer->m_BasicParameters.m_RecurrentToOutputWeights =
        std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToOutputWeights));
    layer->m_BasicParameters.m_ForgetGateBias =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_ForgetGateBias));
    layer->m_BasicParameters.m_CellBias =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellBias));
    layer->m_BasicParameters.m_OutputGateBias =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_OutputGateBias));

    //Lstm Cifg parameters
    if(!descriptor.m_CifgEnabled)
    {
        if(params.m_InputToInputWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Input To Input Weights cannot be NULL");
        }
        if(params.m_RecurrentToInputWeights == nullptr)
        {
            throw InvalidArgumentException(
                    "AddLstmLayer: Recurrent To Input Weights cannot be NULL");
        }
        if(params.m_InputGateBias == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Input Gate Bias cannot be NULL");
        }
        layer->m_CifgParameters.m_InputToInputWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputToInputWeights));
        layer->m_CifgParameters.m_RecurrentToInputWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_RecurrentToInputWeights));
        // In the VTS tests, cell-to-input weights may be null, even if the other CIFG params are not.
        if(params.m_CellToInputWeights != nullptr)
        {
            layer->m_CifgParameters.m_CellToInputWeights =
                    std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToInputWeights));
        }
        layer->m_CifgParameters.m_InputGateBias =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_InputGateBias));
    }

    //Lstm projection parameters
    if(descriptor.m_ProjectionEnabled)
    {
        if(params.m_ProjectionWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Projection Weights cannot be NULL");
        }
        layer->m_ProjectionParameters.m_ProjectionWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_ProjectionWeights));
        if(params.m_ProjectionBias != nullptr)
        {
            layer->m_ProjectionParameters.m_ProjectionBias =
                std::make_unique<ScopedCpuTensorHandle>(*(params.m_ProjectionBias));
        }
    }

    //Lstm Peephole params
    if(descriptor.m_PeepholeEnabled)
    {
        if(params.m_CellToForgetWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell To Forget Weights cannot be NULL");
        }
        if(params.m_CellToOutputWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell To Output Weights cannot be NULL");
        }
        layer->m_PeepholeParameters.m_CellToForgetWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToForgetWeights));
        layer->m_PeepholeParameters.m_CellToOutputWeights =
            std::make_unique<ScopedCpuTensorHandle>(*(params.m_CellToOutputWeights));
    }
    return layer;
}

IConnectableLayer* Network::AddDivisionLayer(const char* name)
{
    return m_Graph->AddLayer<DivisionLayer>(name);
}

IConnectableLayer* Network::AddSubtractionLayer(const char* name)
{
    return m_Graph->AddLayer<SubtractionLayer>(name);
}

IConnectableLayer* Network::AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name)
{
    return m_Graph->AddLayer<MeanLayer>(meanDescriptor,name);
}

OptimizedNetwork::OptimizedNetwork(std::unique_ptr<Graph> graph)
    : m_Graph(std::move(graph))
{
}

OptimizedNetwork::~OptimizedNetwork()
{
}

} // namespace armnn
