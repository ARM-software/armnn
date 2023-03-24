//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Network.hpp"
#include "Graph.hpp"
#include "Layer.hpp"
#include "DeviceSpec.hpp"
#include "Optimizer.hpp"
#include "SubgraphViewSelector.hpp"
#include "BackendSettings.hpp"
#include "optimizations/All.hpp"
#include "armnnUtils/Filesystem.hpp"
#include "armnn/utility/Timer.hpp"

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>

#include <armnn/Exceptions.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <client/include/IProfilingService.hpp>

#include <common/include/ProfilingGuid.hpp>

#include <fmt/format.h>

#include <fcntl.h>
#include <algorithm>
#include <memory>
#include <vector>

namespace armnn
{

INetwork::INetwork(NetworkOptions networkOptions) : pNetworkImpl(new NetworkImpl(networkOptions)) {}

INetwork::~INetwork() = default;

OptimizerOptionsOpaque::OptimizerOptionsOpaque()
        : p_OptimizerOptionsImpl(std::make_unique<OptimizerOptionsOpaqueImpl>())
{
}

OptimizerOptionsOpaque::OptimizerOptionsOpaque(OptimizerOptionsOpaque const &other)
        : p_OptimizerOptionsImpl(std::make_unique<OptimizerOptionsOpaqueImpl>(*other.p_OptimizerOptionsImpl))
{
}

OptimizerOptionsOpaque::~OptimizerOptionsOpaque() = default;

OptimizerOptionsOpaque::OptimizerOptionsOpaque(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16,
                                               bool importEnabled, ModelOptions modelOptions, bool exportEnabled,
                                               bool debugToFile)
        : p_OptimizerOptionsImpl(std::make_unique<OptimizerOptionsOpaqueImpl>(reduceFp32ToFp16, debug, reduceFp32ToBf16,
                                                                              importEnabled, modelOptions,
                                                                              exportEnabled, debugToFile))
{
}

OptimizerOptionsOpaque::OptimizerOptionsOpaque(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16,
                                               ShapeInferenceMethod shapeInferenceMethod,
                                               bool importEnabled, ModelOptions modelOptions, bool exportEnabled,
                                               bool debugToFile, bool allowExpandedDims)
        : p_OptimizerOptionsImpl(std::make_unique<OptimizerOptionsOpaqueImpl>(reduceFp32ToFp16, debug, reduceFp32ToBf16,
                                                                              shapeInferenceMethod, importEnabled,
                                                                              modelOptions, exportEnabled,
                                                                              debugToFile, allowExpandedDims))
{
}

OptimizerOptionsOpaque::OptimizerOptionsOpaque(const OptimizerOptions& OptimizerStruct)
    : p_OptimizerOptionsImpl(std::make_unique<OptimizerOptionsOpaqueImpl>())
{
    p_OptimizerOptionsImpl->m_ImportEnabled = OptimizerStruct.m_ImportEnabled;
    p_OptimizerOptionsImpl->m_shapeInferenceMethod = OptimizerStruct.m_shapeInferenceMethod;
    p_OptimizerOptionsImpl->m_ModelOptions = OptimizerStruct.m_ModelOptions;
    p_OptimizerOptionsImpl->m_ProfilingEnabled = OptimizerStruct.m_ProfilingEnabled;
    p_OptimizerOptionsImpl->m_DebugToFile = OptimizerStruct.m_DebugToFile;
    p_OptimizerOptionsImpl->m_Debug = OptimizerStruct.m_Debug;
    p_OptimizerOptionsImpl->m_ReduceFp32ToFp16 = OptimizerStruct.m_ReduceFp32ToFp16;
    p_OptimizerOptionsImpl->m_ExportEnabled = OptimizerStruct.m_ExportEnabled;
    p_OptimizerOptionsImpl->m_AllowExpandedDims = OptimizerStruct.m_AllowExpandedDims;
    p_OptimizerOptionsImpl->m_ReduceFp32ToBf16 = OptimizerStruct.m_ReduceFp32ToBf16;
}

OptimizerOptionsOpaque& OptimizerOptionsOpaque::operator= (OptimizerOptionsOpaque other)
{
    p_OptimizerOptionsImpl->m_ImportEnabled = other.GetImportEnabled();
    p_OptimizerOptionsImpl->m_shapeInferenceMethod = other.GetShapeInferenceMethod();
    p_OptimizerOptionsImpl->m_ModelOptions = other.GetModelOptions();
    p_OptimizerOptionsImpl->m_ProfilingEnabled = other.GetProfilingEnabled();
    p_OptimizerOptionsImpl->m_DebugToFile = other.GetDebugToFileEnabled();
    p_OptimizerOptionsImpl->m_Debug = other.GetDebugEnabled();
    p_OptimizerOptionsImpl->m_ReduceFp32ToFp16 = other.GetReduceFp32ToFp16();
    p_OptimizerOptionsImpl->m_ExportEnabled = other.GetExportEnabled();
    p_OptimizerOptionsImpl->m_AllowExpandedDims = other.GetAllowExpandedDims();
    p_OptimizerOptionsImpl->m_ReduceFp32ToBf16 = other.GetReduceFp32ToBf16();
    return *this;
}

void OptimizerOptionsOpaque::SetImportEnabled(bool ImportState)
{
    p_OptimizerOptionsImpl->m_ImportEnabled = ImportState;
}

void OptimizerOptionsOpaque::SetExportEnabled(bool ExportState)
{
    p_OptimizerOptionsImpl->m_ExportEnabled = ExportState;
}

void OptimizerOptionsOpaque::SetProfilingEnabled(bool ProfilingState)
{
    p_OptimizerOptionsImpl->m_ProfilingEnabled = ProfilingState;
}

void OptimizerOptionsOpaque::SetDebugEnabled(bool DebugState)
{
    p_OptimizerOptionsImpl->m_Debug = DebugState;
}

void OptimizerOptionsOpaque::SetDebugToFileEnabled(bool DebugFileState)
{
    p_OptimizerOptionsImpl->m_DebugToFile = DebugFileState;
}

void OptimizerOptionsOpaque::SetReduceFp32ToFp16(bool ReduceFp32ToFp16State)
{
    p_OptimizerOptionsImpl->m_ReduceFp32ToFp16 = ReduceFp32ToFp16State;
}

void OptimizerOptionsOpaque::SetShapeInferenceMethod(armnn::ShapeInferenceMethod ShapeInferenceMethodType)
{
    p_OptimizerOptionsImpl->m_shapeInferenceMethod = ShapeInferenceMethodType;
}

void OptimizerOptionsOpaque::SetAllowExpandedDims(bool ExpandedDimsAllowed)
{
    p_OptimizerOptionsImpl->m_AllowExpandedDims = ExpandedDimsAllowed;
}

void OptimizerOptionsOpaque::AddModelOption(armnn::BackendOptions NewModelOption)
{
    p_OptimizerOptionsImpl->m_ModelOptions.push_back(NewModelOption);
}

bool OptimizerOptionsOpaque::GetProfilingEnabled() const
{
    return p_OptimizerOptionsImpl->m_ProfilingEnabled;
};

bool OptimizerOptionsOpaque::GetImportEnabled() const
{
    return p_OptimizerOptionsImpl->m_ImportEnabled;
};

bool OptimizerOptionsOpaque::GetExportEnabled() const
{
    return p_OptimizerOptionsImpl->m_ExportEnabled;
};

bool OptimizerOptionsOpaque::GetReduceFp32ToFp16() const
{
    return p_OptimizerOptionsImpl->m_ReduceFp32ToFp16;
};

bool OptimizerOptionsOpaque::GetReduceFp32ToBf16() const
{
    return p_OptimizerOptionsImpl->m_ReduceFp32ToBf16;
}

bool OptimizerOptionsOpaque::GetDebugEnabled() const
{
    return p_OptimizerOptionsImpl->m_Debug;
}

bool OptimizerOptionsOpaque::GetDebugToFileEnabled() const
{
    return p_OptimizerOptionsImpl->m_DebugToFile;
}

bool OptimizerOptionsOpaque::GetAllowExpandedDims() const
{
    return p_OptimizerOptionsImpl->m_AllowExpandedDims;
}

armnn::ModelOptions OptimizerOptionsOpaque::GetModelOptions() const
{
    return p_OptimizerOptionsImpl->m_ModelOptions;
}

armnn::ShapeInferenceMethod OptimizerOptionsOpaque::GetShapeInferenceMethod() const
{
    return p_OptimizerOptionsImpl->m_shapeInferenceMethod;
}

const std::string OptimizerOptionsOpaque::ToString() const
{
    std::stringstream stream;
    stream << "OptimizerOptions: \n";
    stream << "\tReduceFp32ToFp16: " << p_OptimizerOptionsImpl->m_ReduceFp32ToFp16 << "\n";
    stream << "\tReduceFp32ToBf16: " << p_OptimizerOptionsImpl->m_ReduceFp32ToBf16 << "\n";
    stream << "\tDebug: " << p_OptimizerOptionsImpl->m_Debug << "\n";
    stream << "\tDebug to file: " << p_OptimizerOptionsImpl->m_DebugToFile << "\n";
    stream << "\tShapeInferenceMethod: " <<
           (p_OptimizerOptionsImpl->m_shapeInferenceMethod == ShapeInferenceMethod::ValidateOnly ?
           "ValidateOnly" : "InferAndValidate") << "\n";
    stream << "\tImportEnabled: " << p_OptimizerOptionsImpl->m_ImportEnabled << "\n";
    stream << "\tExportEnabled: " << p_OptimizerOptionsImpl->m_ExportEnabled << "\n";
    stream << "\tProfilingEnabled: " << p_OptimizerOptionsImpl->m_ProfilingEnabled << "\n";
    stream << "\tAllowExpandedDims: " << p_OptimizerOptionsImpl->m_AllowExpandedDims << "\n";

    stream << "\tModelOptions: \n";
    for (auto optionsGroup : p_OptimizerOptionsImpl->m_ModelOptions)
    {
        for (size_t i=0; i < optionsGroup.GetOptionCount(); i++)
        {
            const armnn::BackendOptions::BackendOption option = optionsGroup.GetOption(i);
            stream << "\t\tBackend: "  << optionsGroup.GetBackendId() << "\n"
                   << "\t\t\tOption: " << option.GetName() << "\n"
                   << "\t\t\tValue: "  << std::string(option.GetValue().ToString()) << "\n";
        }
    }

    return stream.str();
}

Status INetwork::PrintGraph()
{
    return pNetworkImpl->PrintGraph();
}

IConnectableLayer* INetwork::AddInputLayer(LayerBindingId id, const char* name)
{
    return pNetworkImpl->AddInputLayer(id, name);
}

IConnectableLayer* INetwork::AddArgMinMaxLayer(const ArgMinMaxDescriptor& desc,
                                               const char* name)
{
    return pNetworkImpl->AddArgMinMaxLayer(desc, name);
}

IConnectableLayer* INetwork::AddCastLayer(const char* name)
{
    return pNetworkImpl->AddCastLayer(name);
}

IConnectableLayer* INetwork::AddComparisonLayer(const ComparisonDescriptor& comparisonDescriptor,
                                                const char* name)
{
    return pNetworkImpl->AddComparisonLayer(comparisonDescriptor, name);
}


IConnectableLayer* INetwork::AddConcatLayer(const ConcatDescriptor& concatDescriptor,
                                            const char* name)
{
    return pNetworkImpl->AddConcatLayer(concatDescriptor, name);
}


IConnectableLayer* INetwork::AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                                   const char* name)
{
    return pNetworkImpl->AddConvolution2dLayer(convolution2dDescriptor, name);
}

IConnectableLayer* INetwork::AddConvolution3dLayer(const Convolution3dDescriptor& convolution3dDescriptor,
                                                   const char* name)
{
    return pNetworkImpl->AddConvolution3dLayer(convolution3dDescriptor, name);
}


IConnectableLayer* INetwork::AddDepthToSpaceLayer(const DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                                  const char* name)
{
    return pNetworkImpl->AddDepthToSpaceLayer(depthToSpaceDescriptor, name);
}


IConnectableLayer* INetwork::AddDepthwiseConvolution2dLayer(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const char* name)
{
    return pNetworkImpl->AddDepthwiseConvolution2dLayer(convolution2dDescriptor, name);
}


IConnectableLayer* INetwork::AddDequantizeLayer(const char* name)
{
    return pNetworkImpl->AddDequantizeLayer(name);
}


IConnectableLayer* INetwork::AddDetectionPostProcessLayer(
    const DetectionPostProcessDescriptor& descriptor,
    const ConstTensor& anchors,
    const char* name)
{
    return pNetworkImpl->AddDetectionPostProcessLayer(descriptor, anchors, name);
}

IConnectableLayer* INetwork::AddElementwiseBinaryLayer(const ElementwiseBinaryDescriptor& elementwiseBinaryDescriptor,
                                                       const char* name)
{
    return pNetworkImpl->AddElementwiseBinaryLayer(elementwiseBinaryDescriptor, name);
}

IConnectableLayer* INetwork::AddElementwiseUnaryLayer(const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                                      const char* name)
{
    return pNetworkImpl->AddElementwiseUnaryLayer(elementwiseUnaryDescriptor, name);
}

IConnectableLayer* INetwork::AddFillLayer(const FillDescriptor& fillDescriptor,
                                          const char* name)
{
    return pNetworkImpl->AddFillLayer(fillDescriptor, name);
}

IConnectableLayer* INetwork::AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                    const char* name)
{
    return pNetworkImpl->AddFullyConnectedLayer(fullyConnectedDescriptor, name);
}

IConnectableLayer* INetwork::AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                             const char* name)
{
    return pNetworkImpl->AddPermuteLayer(permuteDescriptor, name);
}

IConnectableLayer* INetwork::AddBatchToSpaceNdLayer(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                                    const char* name)
{
    return pNetworkImpl->AddBatchToSpaceNdLayer(batchToSpaceNdDescriptor, name);
}

IConnectableLayer* INetwork::AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
                                               const char* name)
{
    return pNetworkImpl->AddPooling2dLayer(pooling2dDescriptor, name);
}

IConnectableLayer* INetwork::AddPooling3dLayer(const Pooling3dDescriptor& pooling3dDescriptor,
                                               const char* name)
{
    return pNetworkImpl->AddPooling3dLayer(pooling3dDescriptor, name);
}

IConnectableLayer* INetwork::AddPrecompiledLayer(const PreCompiledDescriptor& preCompiledDescriptor,
                                                 CompiledBlobPtr compiledBlobPtr,
                                                 const Optional<BackendId>& backend,
                                                 const char* name)
{
    return pNetworkImpl->AddPrecompiledLayer(preCompiledDescriptor, std::move(compiledBlobPtr), backend, name);
}

IConnectableLayer* INetwork::AddActivationLayer(const ActivationDescriptor& activationDescriptor,
                                                const char* name)
{
    return pNetworkImpl->AddActivationLayer(activationDescriptor, name);
}

IConnectableLayer* INetwork::AddNormalizationLayer(const NormalizationDescriptor& normalizationDescriptor,
                                                   const char* name)
{
    return pNetworkImpl->AddNormalizationLayer(normalizationDescriptor, name);
}

IConnectableLayer* INetwork::AddSliceLayer(const SliceDescriptor& sliceDescriptor, const char* name)
{
    return pNetworkImpl->AddSliceLayer(sliceDescriptor, name);
}
IConnectableLayer* INetwork::AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
                                             const char* name)
{
    return pNetworkImpl->AddSoftmaxLayer(softmaxDescriptor, name);
}

IConnectableLayer* INetwork::AddSplitterLayer(const ViewsDescriptor& splitterDescriptor,
                                              const char* name)
{
    return pNetworkImpl->AddSplitterLayer(splitterDescriptor, name);
}

IConnectableLayer* INetwork::AddMergeLayer(const char* name)
{
    return pNetworkImpl->AddMergeLayer(name);
}

IConnectableLayer* INetwork::AddAdditionLayer(const char* name)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return pNetworkImpl->AddAdditionLayer(name);
    ARMNN_NO_DEPRECATE_WARN_END
}

IConnectableLayer* INetwork::AddMultiplicationLayer(const char* name)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return pNetworkImpl->AddMultiplicationLayer(name);
    ARMNN_NO_DEPRECATE_WARN_END
}

IConnectableLayer* INetwork::AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
                                                        const ConstTensor& mean,
                                                        const ConstTensor& variance,
                                                        const ConstTensor& beta,
                                                        const ConstTensor& gamma,
                                                        const char* name)
{
    return pNetworkImpl->AddBatchNormalizationLayer(desc, mean, variance, beta, gamma, name);
}

IConnectableLayer* INetwork::AddRankLayer(const char* name)
{
    return pNetworkImpl->AddRankLayer(name);
}

IConnectableLayer* INetwork::AddResizeLayer(const ResizeDescriptor& resizeDescriptor,
                                            const char* name)
{
    return pNetworkImpl->AddResizeLayer(resizeDescriptor, name);
}

IConnectableLayer* INetwork::AddReduceLayer(const ReduceDescriptor& reduceDescriptor,
                                            const char* name)
{
    return pNetworkImpl->AddReduceLayer(reduceDescriptor, name);
}

IConnectableLayer* INetwork::AddInstanceNormalizationLayer(const InstanceNormalizationDescriptor& desc,
                                                           const char* name)
{
    return pNetworkImpl->AddInstanceNormalizationLayer(desc, name);
}

IConnectableLayer* INetwork::AddL2NormalizationLayer(const L2NormalizationDescriptor& desc,
                                                     const char* name)
{
    return pNetworkImpl->AddL2NormalizationLayer(desc, name);
}

IConnectableLayer* INetwork::AddLogSoftmaxLayer(const LogSoftmaxDescriptor& logSoftmaxDescriptor,
                                                const char* name)
{
    return pNetworkImpl->AddLogSoftmaxLayer(logSoftmaxDescriptor, name);
}

IConnectableLayer* INetwork::AddConstantLayer(const ConstTensor& input,
                                              const char* name)
{
    return pNetworkImpl->AddConstantLayer(input, name);
}

IConnectableLayer* INetwork::AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                            const char* name)
{
    return pNetworkImpl->AddReshapeLayer(reshapeDescriptor, name);
}

IConnectableLayer* INetwork::AddSpaceToBatchNdLayer(const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                   const char* name)
{
    return pNetworkImpl->AddSpaceToBatchNdLayer(spaceToBatchNdDescriptor, name);
}

IConnectableLayer* INetwork::AddSpaceToDepthLayer(const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                                  const char* name)
{
    return pNetworkImpl->AddSpaceToDepthLayer(spaceToDepthDescriptor, name);
}

IConnectableLayer* INetwork::AddFloorLayer(const char* name)
{
    return pNetworkImpl->AddFloorLayer(name);
}
IConnectableLayer* INetwork::AddOutputLayer(LayerBindingId id, const char* name)
{
    return pNetworkImpl->AddOutputLayer(id, name);
}

IConnectableLayer* INetwork::AddLstmLayer(const LstmDescriptor& descriptor,
                                          const LstmInputParams& params,
                                          const char* name)
{
    return pNetworkImpl->AddLstmLayer(descriptor, params, name);
}

IConnectableLayer* INetwork::AddDivisionLayer(const char* name)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return pNetworkImpl->AddDivisionLayer(name);
    ARMNN_NO_DEPRECATE_WARN_END
}

IConnectableLayer* INetwork::AddSubtractionLayer(const char* name)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return pNetworkImpl->AddSubtractionLayer(name);
    ARMNN_NO_DEPRECATE_WARN_END
}

IConnectableLayer* INetwork::AddMaximumLayer(const char* name)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return pNetworkImpl->AddMaximumLayer(name);
    ARMNN_NO_DEPRECATE_WARN_END
}

IConnectableLayer* INetwork::AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name)
{
    return pNetworkImpl->AddMeanLayer(meanDescriptor, name);
}

IConnectableLayer* INetwork::AddPadLayer(const PadDescriptor& padDescriptor,
                                         const char* name)
{
    return pNetworkImpl->AddPadLayer(padDescriptor, name);
}

IConnectableLayer* INetwork::AddQuantizeLayer(const char* name)
{
    return pNetworkImpl->AddQuantizeLayer(name);
}

IConnectableLayer* INetwork::AddStridedSliceLayer(const StridedSliceDescriptor& stridedSliceDescriptor,
                                                  const char* name)
{
    return pNetworkImpl->AddStridedSliceLayer(stridedSliceDescriptor, name);
}

IConnectableLayer* INetwork::AddMinimumLayer(const char* name)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return pNetworkImpl->AddMinimumLayer(name);
    ARMNN_NO_DEPRECATE_WARN_END
}

IConnectableLayer* INetwork::AddGatherLayer(const GatherDescriptor& descriptor,
                                            const char* name)
{
    return pNetworkImpl->AddGatherLayer(descriptor, name);
}

IConnectableLayer* INetwork::AddGatherNdLayer(const char* name)
{
    return pNetworkImpl->AddGatherNdLayer(name);
}

IConnectableLayer* INetwork::AddSwitchLayer(const char* name)
{
    return pNetworkImpl->AddSwitchLayer(name);
}

IConnectableLayer* INetwork::AddPreluLayer(const char* name)
{
    return pNetworkImpl->AddPreluLayer(name);
}

IConnectableLayer* INetwork::AddTransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& descriptor,
                                                            const ConstTensor& weights,
                                                            const Optional<ConstTensor>& biases,
                                                            const char* name)
{
    return pNetworkImpl->AddTransposeConvolution2dLayer(descriptor, weights, biases, name);
}

IConnectableLayer* INetwork::AddTransposeLayer(const TransposeDescriptor& transposeDescriptor,
                                               const char* name)
{
    return pNetworkImpl->AddTransposeLayer(transposeDescriptor, name);
}

IConnectableLayer* INetwork::AddShapeLayer(const char* name)
{
    return pNetworkImpl->AddShapeLayer(name);
}

IConnectableLayer* INetwork::AddStackLayer(const StackDescriptor& descriptor,
                                           const char* name)
{
    return pNetworkImpl->AddStackLayer(descriptor, name);
}

IConnectableLayer* INetwork::AddStandInLayer(const StandInDescriptor& descriptor,
                                             const char* name)
{
    return pNetworkImpl->AddStandInLayer(descriptor, name);
}

IConnectableLayer* INetwork::AddQuantizedLstmLayer(const QuantizedLstmInputParams& params,
                                                   const char* name)
{
    return pNetworkImpl->AddQuantizedLstmLayer(params, name);
}

IConnectableLayer* INetwork::AddQLstmLayer(const QLstmDescriptor& descriptor,
                                           const LstmInputParams& params,
                                           const char* name)
{
    return pNetworkImpl->AddQLstmLayer(descriptor, params, name);
}

IConnectableLayer* INetwork::AddLogicalBinaryLayer(const LogicalBinaryDescriptor& descriptor,
                                                   const char* name)
{
    return pNetworkImpl->AddLogicalBinaryLayer(descriptor, name);
}

IConnectableLayer* INetwork::AddUnidirectionalSequenceLstmLayer(
    const UnidirectionalSequenceLstmDescriptor& descriptor,
    const LstmInputParams& params,
    const char* name)
{
    return pNetworkImpl->AddUnidirectionalSequenceLstmLayer(descriptor, params, name);
}

IConnectableLayer* INetwork::AddChannelShuffleLayer(const ChannelShuffleDescriptor &descriptor,
                                                    const char* name)
{
    return pNetworkImpl->AddChannelShuffleLayer(descriptor, name);
}

IConnectableLayer* INetwork::AddBatchMatMulLayer(const BatchMatMulDescriptor &descriptor,
                                                 const char* name)
{
    return pNetworkImpl->AddBatchMatMulLayer(descriptor, name);
}

void INetwork::ExecuteStrategy(IStrategy& strategy) const
{
    return pNetworkImpl->ExecuteStrategy(strategy);
}

armnn::INetwork* INetwork::CreateRaw(const NetworkOptions& networkOptions)
{
    return new INetwork(networkOptions);
}

armnn::INetworkPtr INetwork::Create(const NetworkOptions& networkOptions)
{
    return INetworkPtr(CreateRaw(networkOptions), &INetwork::Destroy);
}

void INetwork::Destroy(INetwork* network)
{
    delete network;
}

IOptimizedNetwork::IOptimizedNetwork(const IOptimizedNetwork& other, const ModelOptions& modelOptions)
    : pOptimizedNetworkImpl(new OptimizedNetworkImpl(*other.pOptimizedNetworkImpl.get(), modelOptions)) {}

IOptimizedNetwork::IOptimizedNetwork(std::unique_ptr<Graph> graph)
    : pOptimizedNetworkImpl(new OptimizedNetworkImpl(std::move(graph))) {}

IOptimizedNetwork::IOptimizedNetwork(std::unique_ptr<OptimizedNetworkImpl> impl)
    : pOptimizedNetworkImpl(std::move(impl)) {}

IOptimizedNetwork::IOptimizedNetwork(std::unique_ptr<Graph> graph, const ModelOptions& modelOptions)
    : pOptimizedNetworkImpl(new OptimizedNetworkImpl(std::move(graph), modelOptions)) {}

IOptimizedNetwork::~IOptimizedNetwork() = default;

void IOptimizedNetwork::Destroy(IOptimizedNetwork* network)
{
    delete network;
}

Status IOptimizedNetwork::PrintGraph()
{
    return pOptimizedNetworkImpl->PrintGraph();
}

Status IOptimizedNetwork::SerializeToDot(std::ostream& stream) const
{
    return pOptimizedNetworkImpl->SerializeToDot(stream);
}

const std::shared_ptr<IProfiler>& IOptimizedNetwork::GetProfiler() const
{
    return pOptimizedNetworkImpl->GetGraph().GetProfiler();
}

arm::pipe::ProfilingGuid IOptimizedNetwork::GetGuid() const
{
    return pOptimizedNetworkImpl->GetGuid();
}

size_t IOptimizedNetwork::GetNumInputs() const
{
    return pOptimizedNetworkImpl->GetNumInputs();
}

size_t IOptimizedNetwork::GetNumOutputs() const
{
    return pOptimizedNetworkImpl->GetNumOutputs();
}

Status OptimizedNetworkImpl::PrintGraph()
{
    m_Graph->Print();
    return Status::Success;
}

Status OptimizedNetworkImpl::SerializeToDot(std::ostream& stream) const
{
    return m_Graph->SerializeToDot(stream);
}

size_t OptimizedNetworkImpl::GetNumInputs() const
{
    return m_Graph->GetNumInputs();
}

size_t OptimizedNetworkImpl::GetNumOutputs() const
{
    return m_Graph->GetNumOutputs();
}

void ReportError(const std::string& errorMessage,
                 Optional<std::vector<std::string>&> errorMessages)
{
    std::stringstream fullErrorMessage;
    fullErrorMessage << "ERROR: " << errorMessage;
    ARMNN_LOG(warning) << fullErrorMessage.str();
    if (errorMessages)
    {
        errorMessages.value().push_back(fullErrorMessage.str());
    }
}

void ReportWarning(const std::string& warningMessage,
                   Optional<std::vector<std::string>&> warningMessages)
{
    std::stringstream fullWarningMessage;
    fullWarningMessage << "WARNING: " << warningMessage;
    ARMNN_LOG(warning) << fullWarningMessage.str();
    if (warningMessages)
    {
        warningMessages.value().push_back(fullWarningMessage.str());
    }
}

OptimizationResult ReturnWithError(OptimizationResult res,
                                   const Layer* layer,
                                   const BackendSettings& backendSettings,
                                   Optional<std::vector<std::string>&> errMessages)
{
    std::stringstream failureMsg;
    failureMsg << "Layer of type " << GetLayerTypeAsCString(layer->GetType())
               << " is not supported on any preferred backend " << backendSettings.m_PreferredBackends;
    ReportError(failureMsg.str(), errMessages);

    res.m_Error = true;
    return res;
}


bool CheckScaleSetOnQuantizedType(Layer* layer, Optional<std::vector<std::string>&> errMessages)
{
    bool noErrors = true;
    unsigned int numOutputs = layer->GetNumOutputSlots();
    for (unsigned int i = 0; i < numOutputs; i++) {
        OutputSlot& outputSlot = layer->GetOutputSlot(i);
        TensorInfo info = outputSlot.GetTensorInfo();
        if (DataType::QAsymmU8 == info.GetDataType())
        {
            if (0.f == info.GetQuantizationScale())
            {
                noErrors = false;
                std::stringstream ss;
                ss << "output " << i << " of layer " << GetLayerTypeAsCString(layer->GetType())
                   << " (" << layer->GetNameStr() << ") is of type"
                   << " Quantized 8 bit but its scale parameter has not been set";
                ReportError(ss.str(), errMessages);
            }
            // Softmax under QuantisedAsymm8 must always be scale (1.0f/256.0f) and offset 0
            if ((info.GetQuantizationScale() != (1.0f / 256.0f) ||
                 info.GetQuantizationOffset() != 0) &&
                 layer->GetType() == armnn::LayerType::Softmax)
            {
                std::stringstream ss;
                ss << "Quantization parameters for Softmax layer (Scale: " <<
                info.GetQuantizationScale() << " and Offset: " << info.GetQuantizationOffset() <<
                ") are incorrect and have been updated to Scale: 0.00390625 and Offset: 0";
                ARMNN_LOG(warning) << ss.str();
                info.SetQuantizationScale((1.0f /256.0f));
                info.SetQuantizationOffset(0);
                outputSlot.SetTensorInfo(info);
            }
        }
    }
    return noErrors;
}

OptimizationResult AttemptBackendAssignment(BackendSettings& backendSettings,
                                            Graph& graph,
                                            Layer* layer,
                                            BackendId backend,
                                            DataType dataTypeIn,
                                            DataType dataTypeOut,
                                            const std::vector<BackendId>& availablePreferredBackends,
                                            std::string& reasonIfUnsupported,
                                            Optional<std::vector<std::string>&> errMessages)
{
    OptimizationResult result;

    // Helper lambda to compose meaningful error message before returning with error
    auto ReturnError = [&](const Layer* layer)
        {
            return ReturnWithError(result, layer, backendSettings, errMessages);
        };

    // need to set the compute device on the layer
    // before we can check if it is supported
    layer->SetBackendId(backend);

    // To run FP16 operations on CpuAcc we need at least v8.2 architecture. If the available architecture 
    // is older than v8.2, we can check if the operator is supported by changing operator inputs & outputs
    // to be FP32 and inserting convert layers around the FP32 operator.
    bool isLayerSupported = IWorkloadFactory::IsLayerSupported(*layer, EmptyOptional(), reasonIfUnsupported);
    std::string checkStr = "This CPU architecture does not support F16 data type, you need v8.2 or above";
    if (!isLayerSupported ||
        reasonIfUnsupported.find(checkStr) != std::string::npos)
    {
        if (dataTypeIn == DataType::Float16 || dataTypeOut == DataType::Float16)
        {
            if (IWorkloadFactory::IsLayerSupported(*layer, DataType::Float32, reasonIfUnsupported)
                && layer->GetType() != LayerType::ConvertFp32ToFp16
                && layer->GetType() != LayerType::ConvertFp16ToFp32)
            {
                auto ConstantLayerFromFp16ToFp32 = [](Layer& layer)
                {
                    if (layer.GetType() == LayerType::Constant)
                    {
                        ConstantLayer* constantLayer = PolymorphicDowncast<ConstantLayer*>(&layer);

                        auto& info = constantLayer->m_LayerOutput->GetTensorInfo();

                        if (info.GetDataType() == DataType::Float16)
                        {
                            std::vector<float> newValues(info.GetNumElements());

                            armnnUtils::FloatingPointConverter::ConvertFloat16To32(
                                    constantLayer->m_LayerOutput->GetConstTensor<Half>(),
                                    info.GetNumElements(),
                                    newValues.data());

                            TensorInfo newInfo(info);
                            newInfo.SetDataType(DataType::Float32);
                            ConstTensor newInput(newInfo, newValues);
                            constantLayer->m_LayerOutput.reset(new ScopedTensorHandle(newInput));

                            layer.GetOutputSlot(0).SetTensorInfo(newInfo);
                        }
                    }
                };

                bool checkType = false;

                for (auto inputSlot : layer->GetInputSlots())
                {
                    auto connectedOutputSlot = inputSlot.GetConnectedOutputSlot();
                    if (connectedOutputSlot->GetOwningLayer().GetType() == LayerType::Constant)
                    {
                        if (connectedOutputSlot->GetNumConnections() == 1)
                        {
                            checkType = true;
                            ConstantLayerFromFp16ToFp32(connectedOutputSlot->GetOwningLayer());
                        }
                    }
                }

                // Insert FP16 -> FP32 conversion layer before current layer
                std::vector<ConvertFp16ToFp32Layer*> convertFp16ToFp32Layers;
                if (dataTypeIn == DataType::Float16)
                {
                    convertFp16ToFp32Layers =
                            InsertConvertFp16ToFp32LayersBefore(graph, *layer, checkType);
                }

                // Insert FP32 -> FP16 conversion layer after current layer
                std::vector<ConvertFp32ToFp16Layer*> convertFp32ToFp16Layers;
                if (dataTypeOut == DataType::Float16)
                {
                    convertFp32ToFp16Layers =
                        InsertConvertFp32ToFp16LayersAfter(graph, *layer);
                }

                // Assign a supported backend to the newly introduced conversion layers
                auto AssignFirstSupportedBackend = [&](Layer* layer, BackendId preferredBackend)
                    {
                        bool supportedBackendFound = false;
                        std::string reasonIfUnsupported;

                        // Try preferred backend first
                        layer->SetBackendId(preferredBackend);
                        if (IWorkloadFactory::IsLayerSupported(*layer,
                                                               EmptyOptional(),
                                                               reasonIfUnsupported))
                        {
                            supportedBackendFound = true;
                        }
                        else
                        {
                            for (const auto& backend : availablePreferredBackends)
                            {
                                // Skip preferred backend (we already determined that it is not supported)
                                if (backend == preferredBackend)
                                {
                                    continue;
                                }

                                layer->SetBackendId(backend);
                                if (IWorkloadFactory::IsLayerSupported(*layer,
                                                                       EmptyOptional(),
                                                                       reasonIfUnsupported))
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
                        return ReturnError(convertLayer);
                    }
                }

                for (ConvertFp32ToFp16Layer* convertLayer : convertFp32ToFp16Layers)
                {
                    if (!AssignFirstSupportedBackend(convertLayer, backend))
                    {
                        return ReturnError(convertLayer);
                    }
                }

                return result;
            }
        }

        std::stringstream warningMsg;
        warningMsg << "Layer of type " << GetLayerTypeAsCString(layer->GetType())
                   << " is not supported on requested backend " << layer->GetBackendId().Get()
                   << " for input data type " << GetDataTypeName(dataTypeIn)
                   << " and output data type " << GetDataTypeName(dataTypeOut)
                   << " (reason: " << reasonIfUnsupported
                   << "), falling back to the next backend.";
        ReportWarning(warningMsg.str(), errMessages);

        return OptimizationResult(true, false);
    }
    else
    {
        return result;
    }
}

inline std::vector<DataType> GetLayerInOutDatatype(const Layer* layer)
{
    DataType dataTypeIn  = layer->GetNumInputSlots() == 0 ? DataType::Float32 :
                           layer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType();
    DataType dataTypeOut = layer->GetNumOutputSlots() == 0 ? DataType::Float32 :
                           layer->GetOutputSlot(0).GetTensorInfo().GetDataType();
    return {dataTypeIn, dataTypeOut};
}

// Refactor to allow passing the IConnectableLayer* rather than Layer Iterator
// on Graph and SubgraphView which are different types.
void AssignBackendsIConnectable(OptimizedNetworkImpl* optNetObjPtr,
                                IConnectableLayer* it,
                                Optional<std::vector<std::string>&> errMessages,
                                OptimizationResult& result,
                                BackendSettings& backendSettings,
                                std::vector<BackendId>& availablePreferredBackends)
{
    auto ReturnError = [&](const Layer* layer)
    {
        return ReturnWithError(result, layer, backendSettings, errMessages);
    };

    auto layer = PolymorphicDowncast<Layer*>(it);

    if (layer->GetType() == LayerType::Input)
    {
        return;
    }

    std::vector<DataType> inOutDataType = GetLayerInOutDatatype(layer);

    std::string reasonIfUnsupported;
    bool found = false;
    if (!CheckScaleSetOnQuantizedType(layer, errMessages))
    {
        // don't bomb immediately, find all the quantized outputs
        // which haven't had a scale set and report them all back.
        result.m_Error = true;
    }

    // First try assign layer to hint backend
    if (layer->GetBackendHint().has_value() &&
        backendSettings.IsBackendSupported(layer->GetBackendHint().value()) &&
        AttemptBackendAssignment(backendSettings,
                                 optNetObjPtr->GetGraph(),
                                 layer,
                                 layer->GetBackendHint().value(),
                                 inOutDataType[0],
                                 inOutDataType[1],
                                 availablePreferredBackends,
                                 reasonIfUnsupported,
                                 errMessages).IsOk())
    {
        found = true;
        backendSettings.m_SelectedBackends.insert(layer->GetBackendHint().value());
    }
    else
    {
        // Try assign layer to prefered list of backends
        for (const auto& backend : availablePreferredBackends)
        {
            if (layer->GetBackendHint().has_value() &&
                layer->GetBackendHint().value() == backend)
            {
                continue; //Don't re-test the backend hint
            }

            OptimizationResult res = AttemptBackendAssignment(backendSettings,
                                                              optNetObjPtr->GetGraph(),
                                                              layer,
                                                              backend,
                                                              inOutDataType[0],
                                                              inOutDataType[1],
                                                              availablePreferredBackends,
                                                              reasonIfUnsupported,
                                                              errMessages);

            if (res.IsOk())
            {
                found = true;
                backendSettings.m_SelectedBackends.insert(backend);
                break;
            }
            else if (res.IsError())
            {
                result = res;  // Cannot continue.
                // Note: we don't need to log the error as it would already
                // be logged in AttemptBackendAssignment().
            }
            else
            {
                ARMNN_ASSERT_MSG(res.IsWarningOnly(), "OptimizationResult in unexpected state.");
            }
        }
    }

    // If the layer is unsupported by any devices, log and return a null network.
    if (!found)
    {
        // NOTE: if the layer is not an operation queue type AND we have not got CpuRef as a
        //       fallback we should set the compute device on the layer to CpuRef (these are not
        //       available as accelerated operations, or are only available under certain
        //       conditions, currently they comprise MemCopy, Constant, Permute)
        armnn::LayerType layerType = layer->GetType();
        if (!backendSettings.IsCpuRefUsed() && (layerType == armnn::LayerType::MemCopy ||
                                                layerType == armnn::LayerType::Constant ||
                                                layerType == armnn::LayerType::Permute))
        {
            BackendId cpuBackendId(armnn::Compute::CpuRef);
            layer->SetBackendId(cpuBackendId);
            backendSettings.m_SelectedBackends.insert(cpuBackendId);
        }
        else
        {
            result = ReturnError(layer);
        }
    }

}

OptimizationResult AssignBackends(OptimizedNetworkImpl* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  Graph::Iterator& firstLayer,
                                  Graph::Iterator& lastLayer,
                                  Optional<std::vector<std::string>&> errMessages)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Optimizer_AssignBackends");
    OptimizationResult result;

    auto availablePreferredBackends = backendSettings.GetAvailablePreferredBackends();
    if (availablePreferredBackends.empty())
    {
        std::stringstream failureMsg;
        failureMsg << "No preferred backends are available";
        ReportError(failureMsg.str(), errMessages);

        result.m_Error = true;
        return result;
    }

    for (auto it = firstLayer; it != lastLayer; ++it)
    {
        auto layer = PolymorphicDowncast<Layer*>(*it);
        std::vector<DataType> inOutDataType = GetLayerInOutDatatype(layer);

        // In AttemptBackendAssignment() we check:
        //     - if input/output datatypes of the layer are float16
        //     - if the layer is supported with these datatypes
        // If the layer is not supported (failing on ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED() in clframework),
        // we attempt to insert convertion layers either side of the new fp32 layer.
        bool isFloat16 = false;
        for (auto type : inOutDataType)
        {
            if (type == DataType::Float16)
            {
                isFloat16 = true;
                break;
            }
        }

        if (layer->GetBackendId() == "Unknown" || isFloat16)
        {
            AssignBackendsIConnectable(optNetObjPtr,
                                       *it,
                                       errMessages,
                                       result,
                                       backendSettings,
                                       availablePreferredBackends);
        }
    }

    for (auto it = firstLayer; it != lastLayer; ++it)
    {
        auto layer = PolymorphicDowncast<Layer*>(*it);

        if(layer->GetType() == LayerType::Input)
        {
            BackendId connectedBackendId = layer->GetOutputSlot(0).GetConnection(0)->GetOwningLayer().GetBackendId();
            layer->SetBackendId(connectedBackendId);
        }
    }

    return result;
}

OptimizationResult AssignBackends(OptimizedNetworkImpl* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  SubgraphView::IConnectableLayerIterator& firstLayer,
                                  SubgraphView::IConnectableLayerIterator& lastLayer,
                                  Optional<std::vector<std::string>&> errMessages)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Optimizer_AssignBackends");
    OptimizationResult result;

    auto availablePreferredBackends = backendSettings.GetAvailablePreferredBackends();
    if (availablePreferredBackends.empty())
    {
        std::stringstream failureMsg;
        failureMsg << "No preferred backends are available";
        ReportError(failureMsg.str(), errMessages);

        result.m_Error = true;
        return result;
    }

    for (auto it = firstLayer; it != lastLayer; ++it)
    {
        AssignBackendsIConnectable(optNetObjPtr,
                                   *it,
                                   errMessages,
                                   result,
                                   backendSettings,
                                   availablePreferredBackends);
    }

    for (auto it = firstLayer; it != lastLayer; ++it)
    {
        auto layer = PolymorphicDowncast<Layer*>(*it);

        if(layer->GetType() == LayerType::Input)
        {
            BackendId connectedBackendId = layer->GetOutputSlot(0).GetConnection(0)->GetOwningLayer().GetBackendId();
            layer->SetBackendId(connectedBackendId);
        }
    }

    return result;
}

OptimizationResult AssignBackends(OptimizedNetworkImpl* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  SubgraphView& subgraph,
                                  Optional<std::vector<std::string>&> errMessages)
{
    SubgraphView::IConnectableLayerIterator firstLayer = subgraph.beginIConnectable();
    SubgraphView::IConnectableLayerIterator lastLayer  = subgraph.endIConnectable();
    return AssignBackends(optNetObjPtr,
                          backendSettings,
                          firstLayer,
                          lastLayer,
                          errMessages);
}

BackendsMap CreateSupportedBackends(TensorHandleFactoryRegistry& handleFactoryRegistry,
                                    BackendSettings& backendSettings)
{
    BackendsMap backends;
    auto const& backendRegistry = BackendRegistryInstance();
    for (auto&& selectedBackend : backendSettings.m_SupportedBackends)
    {
        auto backendFactory = backendRegistry.GetFactory(selectedBackend);
        auto backendObjPtr = backendFactory();
        ARMNN_ASSERT(backendObjPtr);

        backendObjPtr->RegisterTensorHandleFactories(handleFactoryRegistry);

        backends[backendObjPtr->GetId()] = std::move(backendObjPtr);
    }

    return backends;
}

OptimizationResult ApplyBackendOptimizations(OptimizedNetworkImpl* optNetObjPtr,
                                             BackendSettings& backendSettings,
                                             BackendsMap& backends,
                                             const ModelOptions& modelOptions,
                                             Optional<std::vector<std::string>&> errMessages)
{
    ARMNN_ASSERT(optNetObjPtr);
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Optimizer_ApplyBackendOptimizations")
    OptimizationResult result;

    // Get the optimized graph
    Graph& optGraph = optNetObjPtr->GetGraph();

    // Run backend specific optimizations
    for (auto&& selectedBackend : backendSettings.m_SelectedBackends)
    {
        auto backendObjPtr = backends.find(selectedBackend)->second.get();
        ARMNN_ASSERT(backendObjPtr);

        if (selectedBackend == armnn::Compute::GpuAcc || selectedBackend == armnn::Compute::CpuAcc)
        {
            Optimizer::Pass(optGraph, MakeOptimizations(optimizations::PermuteDepthwiseConv2dWeights()));
            Optimizer::Pass(optGraph, MakeOptimizations(optimizations::FusePermuteIntoConstLayer()));
        }

        // Select sub-graphs based on backend
        SubgraphViewSelector::Subgraphs subgraphs =
                SubgraphViewSelector::SelectSubgraphs(optGraph,
                                                      // Select layers assigned to the requested backend
                                                      [&backendObjPtr](const Layer& layer)
                                                      {

                                                          return layer.GetType() != LayerType::Input &&
                                                                 layer.GetType() != LayerType::Output &&
                                                                 layer.GetBackendId() == backendObjPtr->GetId();
                                                      });
        if (subgraphs.empty())
        {
            // No sub-graphs found, try with next selected backend
            continue;
        }

        // Try to optimize each sub-graph
        for (auto& subgraph : subgraphs)
        {
            // Try to optimize the current sub-graph
            ARMNN_SCOPED_PROFILING_EVENT(backendObjPtr->GetId(), "Optimizer_OptimizeSubgraph");
            OptimizationViews optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraph, modelOptions);
            ARMNN_ASSERT(optimizationViews.Validate(*subgraph));

            // Optimization attempted, check the resulting optimized sub-graph
            for (auto& substitution : optimizationViews.GetSubstitutions())
            {
                // Sub-graph optimized, substitute the sub-graph with the new optimized one in the main optimized graph
                SubgraphView& replacementSubgraph   = substitution.m_ReplacementSubgraph;
                SubgraphView& substitutableSubgraph = substitution.m_SubstitutableSubgraph;
                optGraph.SubstituteSubgraph(substitutableSubgraph, replacementSubgraph);

                // Assign the current backend to the optimized sub-graph
                const SubgraphView::IConnectableLayers& subgraphLayers = replacementSubgraph.GetIConnectableLayers();
                std::for_each(subgraphLayers.begin(), subgraphLayers.end(), [&selectedBackend](IConnectableLayer* l)
                    {
                        ARMNN_ASSERT(l);
                        PolymorphicDowncast<Layer*>(l)->SetBackendId(selectedBackend);
                    });
            }

            if (!optimizationViews.GetFailedSubgraphs().empty())
            {
                std::stringstream warningMsg;
                warningMsg << "Some sub-graph(s) failed to optimized on " << backendObjPtr->GetId() << " backend.";
                ReportWarning(warningMsg.str(), errMessages);

                // Failed to optimize the given sub-graph, re-assign the sub-graph layers to other available backends
                BackendSettings settingsCopy(backendSettings);
                if (!backendObjPtr->GetId().IsCpuRef())
                {
                    // Add the current backend to the list of backends to ignore
                    settingsCopy.m_IgnoredBackends.insert(backendObjPtr->GetId());
                }

                int count=0;
                for (auto& failedSubgraph : optimizationViews.GetFailedSubgraphs())
                {
                    // An error occurred: the optimization was attempted but not performed, try different backends
                    std::stringstream subgraphMsg;
                    subgraphMsg << "Re-assigning backends to " << failedSubgraph.GetIConnectableLayers().size()
                                << " layers inside sub-graph " << count++;
                    ReportWarning(subgraphMsg.str(), errMessages);

                    OptimizationResult reassignmentResult = AssignBackends(optNetObjPtr,
                                                                           settingsCopy,
                                                                           *subgraph,
                                                                           errMessages);
                    if (reassignmentResult.m_Error)
                    {
                        // Failed to re-assign one of the remaining backends to each layer of the sub-graph
                        result.m_Error = true;
                        return result;
                    }
                }
            }
        }
    }

    return result;
}

bool RequiresCopy(ITensorHandleFactory::FactoryId src,
                  ITensorHandleFactory::FactoryId dst,
                  TensorHandleFactoryRegistry& registry)
{
    if (src != dst)
    {
        ITensorHandleFactory* srcFactory = registry.GetFactory(src);
        ITensorHandleFactory* dstFactory = registry.GetFactory(dst);

        if (srcFactory && dstFactory &&
            (srcFactory->GetExportFlags() & dstFactory->GetImportFlags()) != 0)
        {
            return false;
        }
        return true;
    }
    return false;
}

// Find the handle factory for the input layer which results in fewest required copies.
ITensorHandleFactory::FactoryId CalculateSlotOptionForInput(BackendsMap& backends,
                                                            OutputSlot& slot,
                                                            TensorHandleFactoryRegistry& registry,
                                                            bool importEnabled)
{
    Layer& layer = slot.GetOwningLayer();
    ARMNN_ASSERT(layer.GetType() == LayerType::Input);

    // Explicitly select the tensorhandle factory for InputLayer because the rules for it are slightly different. It
    // doesn't matter which backend it is assigned to because they all use the same implementation, which
    // requires Map/Unmap support. This means that, so long as the handle type supports map/unmap semantics, we can
    // select a factory with maximum compatibility with the layers connected to the InputLayer.

    // First ensure the from backends can support the TensorHandeAPI
    auto frmBackend = backends.find(layer.GetBackendId());
    if (frmBackend == backends.end() ||
        !frmBackend->second->SupportsTensorAllocatorAPI())
    {
        return ITensorHandleFactory::LegacyFactoryId;
    }

    // Go through all connections to the output slot and determine the TensorHandleFactory which results in the
    // fewest copies.
    std::map<ITensorHandleFactory::FactoryId, int> factoryScores;
    int topScore = 0;
    ITensorHandleFactory::FactoryId topChoice = ITensorHandleFactory::LegacyFactoryId;

    for (auto&& connection : slot.GetConnections())
    {

        const Layer& connectedLayer = connection->GetOwningLayer();

        auto toBackend = backends.find(connectedLayer.GetBackendId());
        ARMNN_ASSERT_MSG(toBackend != backends.end(), "Backend id not found for the connected layer");

        if (!toBackend->second.get()->SupportsTensorAllocatorAPI())
        {
            // The destination backend does not support the tensor allocator API, move to the next one
            continue;
        }

        auto dstPrefs = toBackend->second.get()->GetHandleFactoryPreferences();
        for (auto&& dst : dstPrefs)
        {
            // Input layers use the mem copy workload or import, so the selected factory must
            // support either the map/unmap API or Import API
            ITensorHandleFactory* factory = registry.GetFactory(dst);
            if (importEnabled && factory->GetImportFlags() == 0)
            {
                continue;
            }
            else if (!importEnabled && !factory->SupportsMapUnmap())
            {
                continue;
            }

            auto it = factoryScores.find(dst);
            if (it == factoryScores.end())
            {
                // Add new score to the table
                factoryScores[dst] = 0;
                if (topChoice == ITensorHandleFactory::LegacyFactoryId)
                {
                    topChoice = dst;
                }
            }
            else
            {
                // Increase the score
                factoryScores[dst]++;

                // Track the best option
                if (factoryScores[dst] > topScore)
                {
                    topScore = factoryScores[dst];
                    topChoice = dst;
                }
            }
        }
    }

    return topChoice;
}

// Find the handle factory for the output layer which results in fewest required copies.
ITensorHandleFactory::FactoryId CalculateSlotOptionForOutput(BackendsMap& backends,
                                                            OutputSlot& slot,
                                                            TensorHandleFactoryRegistry& registry)
{
    IgnoreUnused(backends, slot, registry);
    return ITensorHandleFactory::DeferredFactoryId;
}

// For all handle factories supported on the source backend, we wish to find the one which requires the fewest copies
// when considering all connections.
ITensorHandleFactory::FactoryId CalculateSlotOption(BackendsMap& backends,
                                                    OutputSlot& outputSlot,
                                                    TensorHandleFactoryRegistry& registry,
                                                    bool exportEnabled)
{
    // First ensure the from backends can support the TensorHandeAPI
    Layer& layer = outputSlot.GetOwningLayer();
    auto frmBackend = backends.find(layer.GetBackendId());
    if (frmBackend == backends.end() ||
        !frmBackend->second->SupportsTensorAllocatorAPI())
    {
        return ITensorHandleFactory::LegacyFactoryId;
    }

    bool outputConnection = false;
    for (auto&& connection : outputSlot.GetConnections())
    {
        const Layer& connectedLayer = connection->GetOwningLayer();
        if (connectedLayer.GetType() == LayerType::Output)
        {
            outputConnection = true;
        }
    }

    IBackendInternal* srcBackend = frmBackend->second.get();
    auto srcPrefs = srcBackend->GetHandleFactoryPreferences();

    // Initialize the scores
    std::map<ITensorHandleFactory::FactoryId, int> factoryScores;
    for (auto&& pref : srcPrefs)
    {
        if (exportEnabled)
        {
            ITensorHandleFactory* factory = registry.GetFactory(pref);
            if (outputConnection)
            {
                // Check if this is fallback case
                bool fallbackConnection = false;
                for (auto&& inputSlot : layer.GetInputSlots())
                {
                        if (inputSlot.GetConnectedOutputSlot()->GetOwningLayer().GetBackendId() != layer.GetBackendId())
                        {
                            fallbackConnection = true;
                        }
                }
                if (fallbackConnection)
                {
                    auto factoryCap = factory->GetCapabilities(&layer, &layer, CapabilityClass::FallbackImportDisabled);
                    // Cannot use factory import if fallback import is not supported.
                    if (!factoryCap.empty())
                    {
                        continue;
                    }
                }
                else if (factory->GetExportFlags() == 0)
                {
                    continue;
                }
            }
            if (!outputConnection)
            {
                auto factoryCap = factory->GetCapabilities(&layer, &layer, CapabilityClass::FallbackImportDisabled);
                // Cannot use factory import if fallback import is not supported.
                if (!factoryCap.empty())
                {
                    continue;
                }
            }

        }
        else
        {
            // Only consider factories that support map/unmap
            ITensorHandleFactory* factory = registry.GetFactory(pref);
            if (!factory->SupportsMapUnmap())
            {
                // The current tensor handle factory does not support the map/unmap strategy, move to the next one
                continue;
            }
        }


        auto it = factoryScores.find(pref);
        if (it == factoryScores.end())
        {
            // Add new score to the table
            factoryScores[pref] = 0;
        }
    }

    // Score each handle factory based on how many times it requires copies on the slot connections
    for (auto&& connection : outputSlot.GetConnections())
    {
        const Layer& connectedLayer = connection->GetOwningLayer();

        auto toBackend = backends.find(connectedLayer.GetBackendId());
        ARMNN_ASSERT_MSG(toBackend != backends.end(), "Backend id not found for the connected layer");

        auto dstPrefs = toBackend->second.get()->GetHandleFactoryPreferences();
        for (auto&& src : srcPrefs)
        {
            if (factoryScores.find(src) == factoryScores.end()) // Don't consider excluded factories
            {
                continue;
            }

            for (auto&& dst : dstPrefs)
            {
                if (RequiresCopy(src, dst, registry))
                {
                    // Copy avoided, increase the score
                    factoryScores[src]++;
                    break;
                }
            }
        }
    }

    // Find the lowest score
    int minScore = std::numeric_limits<int>::max();
    for (auto it : factoryScores)
    {
        minScore = std::min(minScore, it.second);
    }

    // Collect factories matching the best(lowest) score
    std::vector<ITensorHandleFactory::FactoryId> optimalFactories;
    for (auto it : factoryScores)
    {
        if (it.second == minScore)
        {
            optimalFactories.push_back(it.first);
        }
    }

    // For all compatible Factories matching the best score, find the preferred one for the current layer.
    for (auto&& srcPref : srcPrefs)
    {
        for (auto&& comp : optimalFactories)
        {
            if (comp == srcPref)
            {
                return comp;
            }
        }
    }

    return ITensorHandleFactory::LegacyFactoryId;
}

EdgeStrategy CalculateEdgeStrategy(BackendsMap& backends,
                                   ITensorHandleFactory::FactoryId srcFactoryId,
                                   const Layer& layer,
                                   const Layer& connectedLayer,
                                   TensorHandleFactoryRegistry& registry,
                                   bool importEnabled)
{
    auto toBackend = backends.find(connectedLayer.GetBackendId());
    ARMNN_ASSERT_MSG(toBackend != backends.end(), "Backend id not found for the connected layer");

    auto dstPrefs = toBackend->second.get()->GetHandleFactoryPreferences();

    // Legacy API check for backward compatibility
    if (srcFactoryId == ITensorHandleFactory::LegacyFactoryId || dstPrefs.empty())
    {
        if (layer.GetBackendId() != connectedLayer.GetBackendId())
        {
            return EdgeStrategy::CopyToTarget;
        }
        else
        {
            return EdgeStrategy::DirectCompatibility;
        }
    }

    // TensorHandleFactory API present, so perform more sophisticated strategies.
    // Dst Output layers don't require copy because they use import or map/unmap
    if (connectedLayer.GetType() == LayerType::Output)
    {
        return EdgeStrategy::DirectCompatibility;
    }

    // Search for direct match in prefs
    for (auto&& pref : dstPrefs)
    {
        if (pref == srcFactoryId)
        {
            return EdgeStrategy::DirectCompatibility;
        }
    }

    // Search for export/import options
    ITensorHandleFactory* srcFactory = registry.GetFactory(srcFactoryId);
    if (srcFactory->GetExportFlags() != 0 && importEnabled)
    {
        for (auto&& pref : dstPrefs)
        {
            ITensorHandleFactory* dstFactory = registry.GetFactory(pref);

            // Handles cases when a destPref is not listed in TensorHandleFactoryRegistry
            if (!dstFactory) {
                continue;
            }
            if ((dstFactory->GetImportFlags() & srcFactory->GetExportFlags()) != 0)
            {
                auto srcCapability = srcFactory->GetCapabilities(&layer, &layer, CapabilityClass::PaddingRequired);
                auto dstCapability = dstFactory->GetCapabilities(&connectedLayer,
                                                                 &connectedLayer,
                                                                 CapabilityClass::PaddingRequired);
                auto srcFallback = srcFactory->GetCapabilities(&layer, &layer, CapabilityClass::FallbackImportDisabled);
                auto dstFallback = dstFactory->GetCapabilities(&connectedLayer,
                                                               &connectedLayer,
                                                               CapabilityClass::FallbackImportDisabled);
                // Do not require memory copy if the source and destination do not require padding.
                if (srcCapability.empty() && dstCapability.empty() && srcFallback.empty() && dstFallback.empty())
                {
                    return EdgeStrategy::ExportToTarget;
                }
            }
        }
    }

    // Search for copy options via map/unmap
    if (srcFactory->SupportsMapUnmap())
    {
        for (auto&& pref : dstPrefs)
        {
            ITensorHandleFactory* dstFactory = registry.GetFactory(pref);
            if (dstFactory && dstFactory->SupportsMapUnmap())
            {
                return EdgeStrategy::CopyToTarget;
            }
        }
    }

    return EdgeStrategy::Undefined;
}

// Select the TensorHandleFactories and the corresponding memory strategy
OptimizationResult SelectTensorHandleStrategy(Graph& optGraph,
                                              BackendsMap& backends,
                                              TensorHandleFactoryRegistry& registry,
                                              bool importEnabled,
                                              bool exportEnabled,
                                              Optional<std::vector<std::string>&> errMessages)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Optimizer_SelectTensorHandleStrategy");
    OptimizationResult result;

    optGraph.ForEachLayer([&backends, &registry, &result, &errMessages, importEnabled, exportEnabled](Layer* layer)
    {
        ARMNN_ASSERT(layer);

        // Lets make sure the backend is in our list of supported backends. Something went wrong during backend
        // assignment if this check fails
        ARMNN_ASSERT(backends.find(layer->GetBackendId()) != backends.end());

        // Check each output separately
        for (unsigned int slotIdx = 0; slotIdx < layer->GetNumOutputSlots(); slotIdx++)
        {
            OutputSlot& outputSlot = layer->GetOutputSlot(slotIdx);

            ITensorHandleFactory::FactoryId slotOption = ITensorHandleFactory::LegacyFactoryId;

            // Calculate the factory to use which results in the fewest copies being made.
            switch(layer->GetType())
            {
                case LayerType::Input:
                    slotOption = CalculateSlotOptionForInput(backends, outputSlot, registry, importEnabled);
                    break;
                case LayerType::Output:
                    slotOption = CalculateSlotOptionForOutput(backends, outputSlot, registry);
                    break;
                default:
                    slotOption = CalculateSlotOption(backends, outputSlot, registry, exportEnabled);
                    break;
            }
            outputSlot.SetTensorHandleFactory(slotOption);

            // Now determine the "best" edge strategy for each connection given the slotOption.
            unsigned int connectionIdx = 0;
            for (auto&& connection : outputSlot.GetConnections())
            {
                const Layer& connectedLayer = connection->GetOwningLayer();

                EdgeStrategy strategy = CalculateEdgeStrategy(backends, slotOption, *layer, connectedLayer,
                                                              registry, importEnabled);

                if (strategy == EdgeStrategy::Undefined)
                {
                    result.m_Error = true;
                    if (errMessages)
                    {
                        errMessages.value().emplace_back("Could not find valid strategy required for compatibility"
                                                         " between backends.");
                    }
                    return;
                }

                outputSlot.SetEdgeStrategy(connectionIdx, strategy);

                connectionIdx++;
            }
        }
    });

    return result;
}

// Forwarding function to remain backward compatible with legacy OptimizerOptions
IOptimizedNetworkPtr Optimize(const Graph& inGraph,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptions& options,
                              Optional<std::vector<std::string>&> messages)
{
    return Optimize(inGraph,
                    backendPreferences,
                    deviceSpec,
                    OptimizerOptionsOpaque(options),
                    messages);
}

IOptimizedNetworkPtr Optimize(const Graph& inGraph,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptionsOpaque& options,
                              Optional<std::vector<std::string>&> messages)
{
    ARMNN_LOG(debug) << options.ToString();

    // Enable profiling
    auto profiler = inGraph.GetProfiler();
    ProfilerManager::GetInstance().RegisterProfiler(profiler.get());
    profiler->EnableProfiling(options.GetProfilingEnabled());

    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Optimizer");
    if (backendPreferences.empty())
    {
        throw InvalidArgumentException("Invoked Optimize with no backends specified");
    }

    if (options.GetReduceFp32ToBf16())
    {
        throw InvalidArgumentException("BFloat16 optimization is currently ignored. In order to use Bf16 optimization "
                                       "Please use the FastMathEnabled backend option for CpuAcc or GpuAcc.");
    }

    if (options.GetReduceFp32ToFp16() && options.GetReduceFp32ToBf16())
    {
        throw InvalidArgumentException("BFloat16 and Float16 optimization cannot be enabled at the same time.");
    }

    // Ensure TensorInfo is set on all output slots of ConstantLayers in the graph
    inGraph.VerifyConstantLayerSetTensorInfo();

    std::unique_ptr<Graph> graph = std::make_unique<Graph>(inGraph);

    // We need to pass on the information about whether import and export is enabled to the LoadNetwork phase.
    // The mechanism to do that is to add model options to the optimized network.
    armnn::BackendOptions importExport("Global",
                                        {{"ImportEnabled", options.GetImportEnabled()},
                                         {"ExportEnabled", options.GetExportEnabled()}});
    ModelOptions optimizedOptions(options.GetModelOptions());
    optimizedOptions.push_back(importExport);

    auto optNet = IOptimizedNetworkPtr(new IOptimizedNetwork(std::move(graph), optimizedOptions),
                                       &IOptimizedNetwork::Destroy);

    IOptimizedNetwork* optNetObjPtr = optNet.get();

    // Get the optimized graph
    Graph& optGraph = optNetObjPtr->pOptimizedNetworkImpl->GetGraph();

    if(options.GetShapeInferenceMethod() == ShapeInferenceMethod::InferAndValidate)
    {
        // Infer the tensor infos for all output slots. Throws an exception on failure
        optGraph.InferTensorInfos();
    }

    // Perform AddBroadcastReshapeLayer optimisation
    using namespace optimizations;
    Optimizer::Pass(optGraph, MakeOptimizations(AddBroadcastReshapeLayer()));

    if(options.GetShapeInferenceMethod() == ShapeInferenceMethod::ValidateOnly)
    {
        // Validate the tensor infos for all output slots. Throws an exception on failure
        optGraph.InferTensorInfos();
    }


    // Group Constant Layer optimizations together where possible.
    // This is important as:
    // FusePermuteIntoConstantLayer must happen before FoldPadIntoDepthwiseConvolution2d and
    // FuseBatchNormIntoDepthwiseConvolution2D.
    // ConvertConstDequantisationLayersToConstLayers must happen before FoldPadIntoConvolution2d
    Optimizer::Pass(optGraph, MakeOptimizations(FusePermuteIntoConstLayer(),
                                                ConvertConstDequantisationLayersToConstLayers()));
    // Perform optimisation passes
    Optimizer::Pass(optGraph, MakeOptimizations(SquashEqualPermuteSiblings(),
                                                SquashEqualTransposeSiblings(),
                                                SquashEqualReshapeSiblings(),
                                                OptimizeInversePermutes(),
                                                OptimizeInverseTransposes(),
                                                MovePermuteUp(),
                                                MoveTransposeUp(),
                                                PermuteAsReshape(),
                                                TransposeAsReshape(),
                                                OptimizeConsecutiveReshapes(),
                                                FoldPadIntoConvolution2d(),
                                                FoldPadIntoDepthwiseConvolution2d(),
                                                FoldPadIntoPooling2d(),
                                                PermuteAndBatchToSpaceAsDepthToSpace(),
                                                TransposeAndBatchToSpaceAsDepthToSpace(),
                                                FuseBatchNormIntoConvolution2DFloat32(),
                                                FuseBatchNormIntoConvolution2DFloat16(),
                                                FuseBatchNormIntoDepthwiseConvolution2DFloat32(),
                                                FuseBatchNormIntoDepthwiseConvolution2DFloat16()));


    if (options.GetReduceFp32ToFp16())
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Optimizer_ReduceFp32ToFp16");
        Optimizer::Pass(optGraph, MakeOptimizations(Fp32NetworkToFp16Converter()));
        Optimizer::Pass(optGraph, MakeOptimizations(ConvertConstantsFloatToHalf()));
    }

    // Initialize backend settings
    BackendSettings backendSettings(backendPreferences, deviceSpec);
    if (backendSettings.GetAvailablePreferredBackends().empty())
    {
        std::stringstream failureMsg;
        failureMsg << "None of the preferred backends " << backendPreferences
                   << " are supported. Current platform provides " << backendSettings.m_SupportedBackends;
        ReportError(failureMsg.str(), messages);
        throw InvalidArgumentException(failureMsg.str());
    }

    // Create a map to temporarily hold initialized backend objects
    TensorHandleFactoryRegistry tensorHandleFactoryRegistry;
    BackendsMap backends = CreateSupportedBackends(tensorHandleFactoryRegistry, backendSettings);

    // Assign an available backend to each layer
    Graph::Iterator firstLayer = optGraph.begin();
    Graph::Iterator lastLayer  = optGraph.end();
    OptimizationResult assignBackendsResult = AssignBackends(optNetObjPtr->pOptimizedNetworkImpl.get(),
                                                             backendSettings,
                                                             firstLayer,
                                                             lastLayer,
                                                             messages);
    if (assignBackendsResult.m_Error)
    {
        // Failed to assign a backend to each layer
        throw InvalidArgumentException("Failed to assign a backend to each layer");
    }

    Optimizer::Pass(optGraph, MakeOptimizations(OptimizeInverseConversionsFp16(),
                                                OptimizeInverseConversionsFp32()));

    // Apply the backend-specific optimizations
    OptimizationResult backendOptimizationResult = ApplyBackendOptimizations(optNetObjPtr->pOptimizedNetworkImpl.get(),
                                                                             backendSettings,
                                                                             backends,
                                                                             options.GetModelOptions(),
                                                                             messages);
    if (backendOptimizationResult.m_Error)
    {
        // Failed to apply the backend-specific optimizations
        throw InvalidArgumentException("Failed to apply the backend-specific optimizations");
    }

    // Convert constants
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Optimizer_ConvertConstants");
        Optimizer::Pass(optGraph, MakeOptimizations(ConvertConstantsFloatToHalf()));
        Optimizer::Pass(optGraph, MakeOptimizations(ConvertConstantsHalfToFloat()));
    }

    // This must occur after all topological changes to the graph and any redirection of variables
    // If the debug flag is set, then insert a DebugLayer after each layer
    // Doing this after applying the backend optimizations as they might have changed some layers
    if (options.GetDebugEnabled() && !options.GetDebugToFileEnabled())
    {
        Optimizer::Pass(optGraph, MakeOptimizations(InsertDebugLayer()));
    }
    else if (options.GetDebugToFileEnabled())
    {
        // Setup the output file path
        try
        {
            auto result = armnnUtils::Filesystem::CreateDirectory("/ArmNNIntermediateLayerOutputs");
            ARMNN_LOG(info) << "Intermediate tensors will be written to: " << result;
            Optimizer::Pass(optGraph, MakeOptimizations(InsertDebugToFileLayer()));
        }
        catch (const armnn::RuntimeException& e)
        {
            // If we cannot create the output directory then we'll issue a warning and continue.
            ARMNN_LOG(warning) << "Unable to print intermediate layer outputs : " << e.what();
        }
    }

    // Calculate the compatibility strategies for tensor handles
    OptimizationResult strategyResult = SelectTensorHandleStrategy(optGraph,
                                                                   backends,
                                                                   tensorHandleFactoryRegistry,
                                                                   options.GetImportEnabled(),
                                                                   options.GetExportEnabled(),
                                                                   messages);

    if (strategyResult.m_Error)
    {
        // Failed to apply the backend-specific optimizations
        return IOptimizedNetworkPtr(nullptr, &IOptimizedNetwork::Destroy);
    }

    // Based on the tensor handle strategy determined above, insert copy layers where required.
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Optimizer_AddCompatibilityLayers");
        optGraph.AddCompatibilityLayers(backends, tensorHandleFactoryRegistry);
    }

    return optNet;
}

// Forwarding function to remain backward compatible with legacy OptimizerOptions
IOptimizedNetworkPtr Optimize(const INetwork& inNetwork,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptions& options,
                              Optional<std::vector<std::string>&> messages)
{
    return Optimize(inNetwork,
                    backendPreferences,
                    deviceSpec,
                    OptimizerOptionsOpaque(options),
                    messages);
}

IOptimizedNetworkPtr Optimize(const INetwork& inNetwork,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptionsOpaque& options,
                              Optional<std::vector<std::string>&> messages)
{
    return Optimize(inNetwork.pNetworkImpl->GetGraph(),
                    backendPreferences,
                    deviceSpec,
                    options,
                    messages);
}

bool NetworkImpl::GetShapeInferenceMethod()
{
    bool shapeInferenceMethod = false;

    ParseOptions(m_NetworkOptions, "ShapeInferenceMethod", [&](std::string name, const BackendOptions::Var& value)
    {
        if (name == "InferAndValidate")
        {
            shapeInferenceMethod |= value.AsBool();
        }
    });
    return shapeInferenceMethod;
}

bool NetworkImpl::GetAllowExpandedDims()
{
    bool allowExpandedDims = false;

    ParseOptions(m_NetworkOptions, "AllowExpandedDims", [&](std::string name, const BackendOptions::Var& value)
    {
        if (name == "AllowExpandedDims")
        {
            allowExpandedDims |= value.AsBool();
        }
    });
    return allowExpandedDims;
}

NetworkImpl::NetworkImpl(const NetworkOptions& networkOptions)
: m_NetworkOptions(networkOptions),
  m_Graph(std::make_unique<Graph>(GetShapeInferenceMethod(), GetAllowExpandedDims()))
{}

NetworkImpl::~NetworkImpl()
{
}

Status NetworkImpl::PrintGraph()
{
    m_Graph->Print();
    return Status::Success;
}

IConnectableLayer* NetworkImpl::AddInputLayer(LayerBindingId id, const char* name)
{
    return m_Graph->AddLayer<InputLayer>(id, name);
}

IConnectableLayer* NetworkImpl::AddBatchToSpaceNdLayer(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                            const char* name)
{
    return m_Graph->AddLayer<BatchToSpaceNdLayer>(batchToSpaceNdDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddCastLayer(const char* name)
{
    return m_Graph->AddLayer<CastLayer>(name);
}
IConnectableLayer* NetworkImpl::AddChannelShuffleLayer(const ChannelShuffleDescriptor& channelShuffleDescriptor,
                                               const char* name)
{
    return m_Graph->AddLayer<ChannelShuffleLayer>(channelShuffleDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddComparisonLayer(const ComparisonDescriptor& comparisonDescriptor,
                                               const char* name)
{
    return m_Graph->AddLayer<ComparisonLayer>(comparisonDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddElementwiseBinaryLayer(const ElementwiseBinaryDescriptor& elementwiseBinaryDesc,
                                                          const char* name)
{
    return m_Graph->AddLayer<ElementwiseBinaryLayer>(elementwiseBinaryDesc, name);
}

IConnectableLayer* NetworkImpl::AddElementwiseUnaryLayer(const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                                     const char* name)
{
    return m_Graph->AddLayer<ElementwiseUnaryLayer>(elementwiseUnaryDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddFillLayer(const FillDescriptor& fillDescriptor,
                                         const char* name)
{
    return m_Graph->AddLayer<FillLayer>(fillDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                       const char* name)
{
    return m_Graph->AddLayer<FullyConnectedLayer>(fullyConnectedDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddConcatLayer(const ConcatDescriptor& concatDescriptor,
                                           const char* name)
{
    return m_Graph->AddLayer<ConcatLayer>(concatDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                                      const char* name)
{
    return m_Graph->AddLayer<Convolution2dLayer>(convolution2dDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddConvertFp16ToFp32Layer(const char* name)
{
    return m_Graph->AddLayer<ConvertFp16ToFp32Layer>(name);
}

IConnectableLayer* NetworkImpl::AddConvertFp32ToFp16Layer(const char* name)
{
    return m_Graph->AddLayer<ConvertFp32ToFp16Layer>(name);
}

IConnectableLayer* NetworkImpl::AddConvolution3dLayer(const Convolution3dDescriptor& convolution3dDescriptor,
                                                      const char* name)
{
    return m_Graph->AddLayer<Convolution3dLayer>(convolution3dDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddDepthToSpaceLayer(const DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                                 const char* name)
{
    return m_Graph->AddLayer<DepthToSpaceLayer>(depthToSpaceDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddDepthwiseConvolution2dLayer(
    const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<DepthwiseConvolution2dLayer>(convolution2dDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddDetectionPostProcessLayer(const armnn::DetectionPostProcessDescriptor& descriptor,
                                                         const ConstTensor& anchors, const char* name)
{
    const auto layer = m_Graph->AddLayer<DetectionPostProcessLayer>(descriptor, name);

    layer->m_Anchors = std::make_shared<ScopedTensorHandle>(anchors);

    return layer;
}

IConnectableLayer* NetworkImpl::AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                            const char* name)
{
    return m_Graph->AddLayer<PermuteLayer>(permuteDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<Pooling2dLayer>(pooling2dDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddPooling3dLayer(const Pooling3dDescriptor& pooling3dDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<Pooling3dLayer>(pooling3dDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddActivationLayer(const ActivationDescriptor& activationDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<ActivationLayer>(activationDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddArgMinMaxLayer(const ArgMinMaxDescriptor& argMinMaxDescriptor,
                                              const char* name)
{
    return m_Graph->AddLayer<ArgMinMaxLayer>(argMinMaxDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddNormalizationLayer(const NormalizationDescriptor&
normalizationDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<NormalizationLayer>(normalizationDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddSliceLayer(const SliceDescriptor& sliceDescriptor, const char* name)
{
    return m_Graph->AddLayer<SliceLayer>(sliceDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<SoftmaxLayer>(softmaxDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddSplitterLayer(const ViewsDescriptor& splitterDescriptor,
    const char* name)
{
    return m_Graph->AddLayer<SplitterLayer>(splitterDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddMaximumLayer(const char* name)
{
    return m_Graph->AddLayer<MaximumLayer>(name);
}

IConnectableLayer* NetworkImpl::AddMinimumLayer(const char* name)
{
    return m_Graph->AddLayer<MinimumLayer>(name);
}

IConnectableLayer* NetworkImpl::AddAdditionLayer(const char* name)
{
    return m_Graph->AddLayer<AdditionLayer>(name);
}

IConnectableLayer* NetworkImpl::AddMultiplicationLayer(const char* name)
{
    return m_Graph->AddLayer<MultiplicationLayer>(name);
}

IConnectableLayer* NetworkImpl::AddOutputLayer(LayerBindingId id, const char* name)
{
    return m_Graph->AddLayer<OutputLayer>(id, name);
}

IConnectableLayer* NetworkImpl::AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
                                                       const ConstTensor&                  mean,
                                                       const ConstTensor&                  variance,
                                                       const ConstTensor&                  beta,
                                                       const ConstTensor&                  gamma,
                                                       const char*                         name)
{
    const auto layer = m_Graph->AddLayer<BatchNormalizationLayer>(desc, name);

    layer->m_Mean = std::make_shared<ScopedTensorHandle>(mean);
    layer->m_Variance = std::make_shared<ScopedTensorHandle>(variance);
    layer->m_Beta = std::make_shared<ScopedTensorHandle>(beta);
    layer->m_Gamma = std::make_shared<ScopedTensorHandle>(gamma);

    return layer;
}

IConnectableLayer* NetworkImpl::AddRankLayer(const char* name)
{
    return m_Graph->AddLayer<RankLayer>(name);
}

IConnectableLayer* NetworkImpl::AddReduceLayer(const ReduceDescriptor& reduceDescriptor,
                                               const char* name)
{
    return m_Graph->AddLayer<ReduceLayer>(reduceDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddResizeLayer(const ResizeDescriptor& resizeDescriptor, const char* name)
{
    return m_Graph->AddLayer<ResizeLayer>(resizeDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddShapeLayer(const char* name)
{
    return m_Graph->AddLayer<ShapeLayer>(name);
}

IConnectableLayer* NetworkImpl::AddInstanceNormalizationLayer(const InstanceNormalizationDescriptor& desc,
                                                              const char* name)
{
    return m_Graph->AddLayer<InstanceNormalizationLayer>(desc, name);
}

IConnectableLayer* NetworkImpl::AddL2NormalizationLayer(const L2NormalizationDescriptor& desc,
                                                        const char* name)
{
    return m_Graph->AddLayer<L2NormalizationLayer>(desc, name);
}

IConnectableLayer* NetworkImpl::AddLogSoftmaxLayer(const LogSoftmaxDescriptor& desc,
                                               const char* name)
{
    return m_Graph->AddLayer<LogSoftmaxLayer>(desc, name);
}

IConnectableLayer* NetworkImpl::AddConstantLayer(const ConstTensor& input, const char* name)
{
    auto layer = m_Graph->AddLayer<ConstantLayer>(name);

    layer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(input);

    return layer;
}

IConnectableLayer* NetworkImpl::AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                            const char* name)
{
    return m_Graph->AddLayer<ReshapeLayer>(reshapeDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddSpaceToBatchNdLayer(const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                   const char* name)
{
    return m_Graph->AddLayer<SpaceToBatchNdLayer>(spaceToBatchNdDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddSpaceToDepthLayer(const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                                 const char* name)
{
    return m_Graph->AddLayer<SpaceToDepthLayer>(spaceToDepthDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddFloorLayer(const char* name)
{
    return m_Graph->AddLayer<FloorLayer>(name);
}

IConnectableLayer* NetworkImpl::AddLstmLayer(const LstmDescriptor&  descriptor,
                                         const LstmInputParams& params,
                                         const char* name)
{
    const auto layer = m_Graph->AddLayer<LstmLayer>(descriptor, name);

    //Lstm Basic Parameters
    layer->m_BasicParameters.m_InputToForgetWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_InputToForgetWeights));
    layer->m_BasicParameters.m_InputToCellWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_InputToCellWeights));
    layer->m_BasicParameters.m_InputToOutputWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_InputToOutputWeights));
    layer->m_BasicParameters.m_RecurrentToForgetWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToForgetWeights));
    layer->m_BasicParameters.m_RecurrentToCellWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToCellWeights));
    layer->m_BasicParameters.m_RecurrentToOutputWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToOutputWeights));
    layer->m_BasicParameters.m_ForgetGateBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_ForgetGateBias));
    layer->m_BasicParameters.m_CellBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_CellBias));
    layer->m_BasicParameters.m_OutputGateBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_OutputGateBias));

    //Lstm Cifg parameters
    if(!descriptor.m_CifgEnabled)
    {
        if(params.m_InputToInputWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Input To Input Weights cannot be NULL "
                                           "when CIFG is disabled.");
        }
        if(params.m_RecurrentToInputWeights == nullptr)
        {
            throw InvalidArgumentException(
                    "AddLstmLayer: Recurrent To Input Weights cannot be NULL "
                    "when CIFG is disabled.");
        }
        if(params.m_InputGateBias == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Input Gate Bias cannot be NULL "
                                           "when CIFG is disabled.");
        }
        layer->m_CifgParameters.m_InputToInputWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_InputToInputWeights));
        layer->m_CifgParameters.m_RecurrentToInputWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToInputWeights));
        layer->m_CifgParameters.m_InputGateBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_InputGateBias));
    }

    //Lstm projection parameters
    if(descriptor.m_ProjectionEnabled)
    {
        if(params.m_ProjectionWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Projection Weights cannot be NULL "
                                           "when projection is enabled.");
        }
        layer->m_ProjectionParameters.m_ProjectionWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_ProjectionWeights));
        if(params.m_ProjectionBias != nullptr)
        {
            layer->m_ProjectionParameters.m_ProjectionBias =
                std::make_shared<ScopedTensorHandle>(*(params.m_ProjectionBias));
        }
    }

    //Lstm Peephole params
    if(descriptor.m_PeepholeEnabled)
    {
        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_CellToInputWeights == nullptr)
            {
                throw InvalidArgumentException("AddLstmLayer: Cell To Input Weights cannot be NULL "
                                               "when Peephole is enabled and CIFG disabled.");
            }

            layer->m_PeepholeParameters.m_CellToInputWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_CellToInputWeights));
        }

        if(params.m_CellToForgetWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell To Forget Weights cannot be NULL "
                                           "when Peephole is enabled.");
        }
        if(params.m_CellToOutputWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell To Output Weights cannot be NULL "
                                           "when Peephole is enabled.");
        }

        layer->m_PeepholeParameters.m_CellToForgetWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_CellToForgetWeights));
        layer->m_PeepholeParameters.m_CellToOutputWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_CellToOutputWeights));
    }

    //Lstm Layer Normalization params
    if(descriptor.m_LayerNormEnabled)
    {
        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_InputLayerNormWeights == nullptr)
            {
                throw InvalidArgumentException("AddLstmLayer: Input layer normalization weights cannot be NULL "
                                               "when layer normalization is enabled and CIFG disabled.");
            }
            layer->m_LayerNormParameters.m_InputLayerNormWeights =
                    std::make_shared<ScopedTensorHandle>(*(params.m_InputLayerNormWeights));
        }

        if(params.m_ForgetLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Forget layer normalization weights cannot be NULL "
                                           "when layer normalization is enabled.");
        }
        if(params.m_CellLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Cell layer normalization weights cannot be NULL "
                                           "when layer normalization is enabled.");
        }
        if(params.m_OutputLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddLstmLayer: Output layer normalization weights cannot be NULL "
                                           "when layer normalization is enabled.");
        }
        layer->m_LayerNormParameters.m_ForgetLayerNormWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_ForgetLayerNormWeights));
        layer->m_LayerNormParameters.m_CellLayerNormWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_CellLayerNormWeights));
        layer->m_LayerNormParameters.m_OutputLayerNormWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_OutputLayerNormWeights));
    }
    return layer;
}

IConnectableLayer* NetworkImpl::AddDivisionLayer(const char* name)
{
    return m_Graph->AddLayer<DivisionLayer>(name);
}

IConnectableLayer* NetworkImpl::AddSubtractionLayer(const char* name)
{
    return m_Graph->AddLayer<SubtractionLayer>(name);
}

IConnectableLayer* NetworkImpl::AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name)
{
    return m_Graph->AddLayer<MeanLayer>(meanDescriptor,name);
}

IConnectableLayer* NetworkImpl::AddPadLayer(const PadDescriptor& padDescriptor, const char* name)
{
    return m_Graph->AddLayer<PadLayer>(padDescriptor,name);
}

IConnectableLayer *NetworkImpl::AddQuantizeLayer(const char *name)
{
    return m_Graph->AddLayer<QuantizeLayer>(name);
}

IConnectableLayer* NetworkImpl::AddDequantizeLayer(const char* name)
{
    return m_Graph->AddLayer<DequantizeLayer>(name);
}

IConnectableLayer* NetworkImpl::AddStridedSliceLayer(const StridedSliceDescriptor& stridedSliceDescriptor,
                                                     const char* name)
{
    return m_Graph->AddLayer<StridedSliceLayer>(stridedSliceDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddGatherLayer(const GatherDescriptor& gatherDescriptor,
                                               const char* name)
{
    return m_Graph->AddLayer<GatherLayer>(gatherDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddGatherNdLayer(const char* name)
{
    return m_Graph->AddLayer<GatherNdLayer>(name);
}

IConnectableLayer* NetworkImpl::AddMergeLayer(const char* name)
{
    return m_Graph->AddLayer<MergeLayer>(name);
}

IConnectableLayer* NetworkImpl::AddSwitchLayer(const char* name)
{
    return m_Graph->AddLayer<SwitchLayer>(name);
}

IConnectableLayer* NetworkImpl::AddPreluLayer(const char* name)
{
    return m_Graph->AddLayer<PreluLayer>(name);
}

IConnectableLayer* NetworkImpl::AddTransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& descriptor,
                                                           const ConstTensor& weights,
                                                           const Optional<ConstTensor>& biases,
                                                           const char* name)
{
    if (descriptor.m_BiasEnabled && !biases.has_value())
    {
        throw InvalidArgumentException("AddTransposeConvolution2dLayer: Biases cannot be empty");
    }

    const auto layer = m_Graph->AddLayer<TransposeConvolution2dLayer>(descriptor, name);

    layer->m_Weight = std::make_shared<ScopedTensorHandle>(weights);

    if (descriptor.m_BiasEnabled)
    {
        layer->m_Bias = std::make_shared<ScopedTensorHandle>(biases.value());
    }

    return layer;
}

IConnectableLayer* NetworkImpl::AddTransposeLayer(const TransposeDescriptor& transposeDescriptor,
                                              const char* name)
{
    return m_Graph->AddLayer<TransposeLayer>(transposeDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddStackLayer(const StackDescriptor& stackDescriptor,
                                          const char* name)
{
    return m_Graph->AddLayer<StackLayer>(stackDescriptor, name);
}


IConnectableLayer* NetworkImpl::AddStandInLayer(const StandInDescriptor& desc,
                                            const char* name)
{
    return m_Graph->AddLayer<StandInLayer>(desc, name);
}

IConnectableLayer* NetworkImpl::AddQuantizedLstmLayer(const QuantizedLstmInputParams& params,
                                                  const char* name)
{
    const auto layer = m_Graph->AddLayer<QuantizedLstmLayer>(name);

    // InputToX weights
    layer->m_QuantizedLstmParameters.m_InputToInputWeights =
            std::make_shared<ScopedTensorHandle>(params.GetInputToInputWeights());
    layer->m_QuantizedLstmParameters.m_InputToForgetWeights =
            std::make_shared<ScopedTensorHandle>(params.GetInputToForgetWeights());
    layer->m_QuantizedLstmParameters.m_InputToCellWeights =
            std::make_shared<ScopedTensorHandle>(params.GetInputToCellWeights());
    layer->m_QuantizedLstmParameters.m_InputToOutputWeights =
            std::make_shared<ScopedTensorHandle>(params.GetInputToOutputWeights());

    // RecurrentToX weights
    layer->m_QuantizedLstmParameters.m_RecurrentToInputWeights =
            std::make_shared<ScopedTensorHandle>(params.GetRecurrentToInputWeights());
    layer->m_QuantizedLstmParameters.m_RecurrentToForgetWeights =
            std::make_shared<ScopedTensorHandle>(params.GetRecurrentToForgetWeights());
    layer->m_QuantizedLstmParameters.m_RecurrentToCellWeights =
            std::make_shared<ScopedTensorHandle>(params.GetRecurrentToCellWeights());
    layer->m_QuantizedLstmParameters.m_RecurrentToOutputWeights =
            std::make_shared<ScopedTensorHandle>(params.GetRecurrentToOutputWeights());

    // Bias
    layer->m_QuantizedLstmParameters.m_InputGateBias =
            std::make_shared<ScopedTensorHandle>(params.GetInputGateBias());
    layer->m_QuantizedLstmParameters.m_ForgetGateBias =
            std::make_shared<ScopedTensorHandle>(params.GetForgetGateBias());
    layer->m_QuantizedLstmParameters.m_CellBias =
            std::make_shared<ScopedTensorHandle>(params.GetCellBias());
    layer->m_QuantizedLstmParameters.m_OutputGateBias =
            std::make_shared<ScopedTensorHandle>(params.GetOutputGateBias());

    return layer;
}

IConnectableLayer* NetworkImpl::AddQLstmLayer(const QLstmDescriptor&  descriptor,
                                          const LstmInputParams& params,
                                          const char* name)
{
    const auto layer = m_Graph->AddLayer<QLstmLayer>(descriptor, name);

    // QLstm Basic Parameters
    layer->m_BasicParameters.m_InputToForgetWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_InputToForgetWeights));
    layer->m_BasicParameters.m_InputToCellWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_InputToCellWeights));
    layer->m_BasicParameters.m_InputToOutputWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_InputToOutputWeights));
    layer->m_BasicParameters.m_RecurrentToForgetWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToForgetWeights));
    layer->m_BasicParameters.m_RecurrentToCellWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToCellWeights));
    layer->m_BasicParameters.m_RecurrentToOutputWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToOutputWeights));
    layer->m_BasicParameters.m_ForgetGateBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_ForgetGateBias));
    layer->m_BasicParameters.m_CellBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_CellBias));
    layer->m_BasicParameters.m_OutputGateBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_OutputGateBias));

    // QLstm Cifg parameters
    if(!descriptor.m_CifgEnabled)
    {
        if(params.m_InputToInputWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Input To Input Weights cannot be NULL");
        }

        if(params.m_RecurrentToInputWeights == nullptr)
        {
            throw InvalidArgumentException(
                    "AddQLstmLayer: Recurrent To Input Weights cannot be NULL");
        }

        if(params.m_InputGateBias == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Input Gate Bias cannot be NULL");
        }

        layer->m_CifgParameters.m_InputToInputWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_InputToInputWeights));
        layer->m_CifgParameters.m_RecurrentToInputWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToInputWeights));
        layer->m_CifgParameters.m_InputGateBias =
                std::make_shared<ScopedTensorHandle>(*(params.m_InputGateBias));
    }

    // QLstm Projection parameters
    if(descriptor.m_ProjectionEnabled)
    {
        if(params.m_ProjectionWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Projection Weights cannot be NULL");
        }

        layer->m_ProjectionParameters.m_ProjectionWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_ProjectionWeights));

        // Projection bias is optional even if projection is enabled
        if(params.m_ProjectionBias != nullptr)
        {
            layer->m_ProjectionParameters.m_ProjectionBias =
                    std::make_shared<ScopedTensorHandle>(*(params.m_ProjectionBias));
        }

    }

    // QLstm Peephole params
    if(descriptor.m_PeepholeEnabled)
    {
        if(params.m_CellToForgetWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Cell To Forget Weights cannot be NULL");
        }

        if(params.m_CellToOutputWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Cell To Output Weights cannot be NULL");
        }

        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_CellToInputWeights == nullptr)
            {
                throw InvalidArgumentException("AddQLstmLayer: Cell To Input Weights cannot be NULL");
            }

            layer->m_PeepholeParameters.m_CellToInputWeights =
                    std::make_shared<ScopedTensorHandle>(*(params.m_CellToInputWeights));
        }

        layer->m_PeepholeParameters.m_CellToForgetWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_CellToForgetWeights));
        layer->m_PeepholeParameters.m_CellToOutputWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_CellToOutputWeights));
    }

    // QLstm Layer Normalization params
    if(descriptor.m_LayerNormEnabled)
    {
        if(params.m_ForgetLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Forget layer normalization weights cannot be NULL");
        }

        if(params.m_CellLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Cell layer normalization weights cannot be NULL");
        }

        if(params.m_OutputLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddQLstmLayer: Output layer normalization weights cannot be NULL");
        }

        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_InputLayerNormWeights == nullptr)
            {
                throw InvalidArgumentException("AddQLstmLayer: Input layer normalization weights cannot be NULL");
            }

            layer->m_LayerNormParameters.m_InputLayerNormWeights =
                    std::make_shared<ScopedTensorHandle>(*(params.m_InputLayerNormWeights));
        }

        layer->m_LayerNormParameters.m_ForgetLayerNormWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_ForgetLayerNormWeights));
        layer->m_LayerNormParameters.m_CellLayerNormWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_CellLayerNormWeights));
        layer->m_LayerNormParameters.m_OutputLayerNormWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_OutputLayerNormWeights));
    }
    return layer;
}

IConnectableLayer* NetworkImpl::AddLogicalBinaryLayer(const LogicalBinaryDescriptor& logicalBinaryDescriptor,
                                                      const char* name)
{
    return m_Graph->AddLayer<LogicalBinaryLayer>(logicalBinaryDescriptor, name);
}

IConnectableLayer* NetworkImpl::AddUnidirectionalSequenceLstmLayer(
    const UnidirectionalSequenceLstmDescriptor&  descriptor,
    const LstmInputParams& params,
    const char* name)
{
    const auto layer = m_Graph->AddLayer<UnidirectionalSequenceLstmLayer>(descriptor, name);

    //Lstm Basic Parameters
    layer->m_BasicParameters.m_InputToForgetWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_InputToForgetWeights));
    layer->m_BasicParameters.m_InputToCellWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_InputToCellWeights));
    layer->m_BasicParameters.m_InputToOutputWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_InputToOutputWeights));
    layer->m_BasicParameters.m_RecurrentToForgetWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToForgetWeights));
    layer->m_BasicParameters.m_RecurrentToCellWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToCellWeights));
    layer->m_BasicParameters.m_RecurrentToOutputWeights =
        std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToOutputWeights));
    layer->m_BasicParameters.m_ForgetGateBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_ForgetGateBias));
    layer->m_BasicParameters.m_CellBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_CellBias));
    layer->m_BasicParameters.m_OutputGateBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_OutputGateBias));

    //Lstm Cifg parameters
    if(!descriptor.m_CifgEnabled)
    {
        if(params.m_InputToInputWeights == nullptr)
        {
            throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Input To Input Weights cannot be NULL "
                                           "when CIFG is disabled.");
        }
        if(params.m_RecurrentToInputWeights == nullptr)
        {
            throw InvalidArgumentException(
                    "AddUnidirectionalSequenceLstmLayer: Recurrent To Input Weights cannot be NULL "
                    "when CIFG is disabled.");
        }
        if(params.m_InputGateBias == nullptr)
        {
            throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Input Gate Bias cannot be NULL "
                                           "when CIFG is disabled.");
        }
        layer->m_CifgParameters.m_InputToInputWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_InputToInputWeights));
        layer->m_CifgParameters.m_RecurrentToInputWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_RecurrentToInputWeights));
        layer->m_CifgParameters.m_InputGateBias =
            std::make_shared<ScopedTensorHandle>(*(params.m_InputGateBias));
    }

    //Lstm projection parameters
    if(descriptor.m_ProjectionEnabled)
    {
        if(params.m_ProjectionWeights == nullptr)
        {
            throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Projection Weights cannot be NULL "
                                           "when projection is enabled.");
        }
        layer->m_ProjectionParameters.m_ProjectionWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_ProjectionWeights));
        if(params.m_ProjectionBias != nullptr)
        {
            layer->m_ProjectionParameters.m_ProjectionBias =
                std::make_shared<ScopedTensorHandle>(*(params.m_ProjectionBias));
        }
    }

    //Lstm Peephole params
    if(descriptor.m_PeepholeEnabled)
    {
        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_CellToInputWeights == nullptr)
            {
                throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Cell To Input Weights "
                                               "cannot be NULL when Peephole is enabled and CIFG disabled.");
            }

            layer->m_PeepholeParameters.m_CellToInputWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_CellToInputWeights));
        }

        if(params.m_CellToForgetWeights == nullptr)
        {
            throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Cell To Forget Weights cannot be NULL "
                                           "when Peephole is enabled.");
        }
        if(params.m_CellToOutputWeights == nullptr)
        {
            throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Cell To Output Weights cannot be NULL "
                                           "when Peephole is enabled.");
        }

        layer->m_PeepholeParameters.m_CellToForgetWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_CellToForgetWeights));
        layer->m_PeepholeParameters.m_CellToOutputWeights =
            std::make_shared<ScopedTensorHandle>(*(params.m_CellToOutputWeights));
    }

    //Lstm Layer Normalization params
    if(descriptor.m_LayerNormEnabled)
    {
        if(!descriptor.m_CifgEnabled)
        {
            if(params.m_InputLayerNormWeights == nullptr)
            {
                throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Input layer normalization weights "
                                               "cannot be NULL when layer normalization is enabled and CIFG disabled.");
            }
            layer->m_LayerNormParameters.m_InputLayerNormWeights =
                    std::make_shared<ScopedTensorHandle>(*(params.m_InputLayerNormWeights));
        }

        if(params.m_ForgetLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Forget layer normalization weights "
                                           "cannot be NULL when layer normalization is enabled.");
        }
        if(params.m_CellLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Cell layer normalization weights "
                                           "cannot be NULL when layer normalization is enabled.");
        }
        if(params.m_OutputLayerNormWeights == nullptr)
        {
            throw InvalidArgumentException("AddUnidirectionalSequenceLstmLayer: Output layer normalization weights "
                                           "cannot be NULL when layer normalization is enabled.");
        }
        layer->m_LayerNormParameters.m_ForgetLayerNormWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_ForgetLayerNormWeights));
        layer->m_LayerNormParameters.m_CellLayerNormWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_CellLayerNormWeights));
        layer->m_LayerNormParameters.m_OutputLayerNormWeights =
                std::make_shared<ScopedTensorHandle>(*(params.m_OutputLayerNormWeights));
    }
    return layer;
}

IConnectableLayer* NetworkImpl::AddBatchMatMulLayer(const BatchMatMulDescriptor& desc, const char* name)
{
    return m_Graph->AddLayer<BatchMatMulLayer>(desc, name);
}

IConnectableLayer* NetworkImpl::AddPrecompiledLayer(const PreCompiledDescriptor& preCompiledDescriptor,
                                                    CompiledBlobPtr compiledBlobPtr,
                                                    const Optional<BackendId>& backend,
                                                    const char* name)
{
    // Method use is for backend users.
    PreCompiledLayer* layer;
    if (name)
    {
        layer = m_Graph->AddLayer<PreCompiledLayer>(preCompiledDescriptor, name);
    }
    else
    {
        layer = m_Graph->AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");
    }

    // Assign the pre-compiled object to layer
    // Pass only one compiled network, Arm NN does not handle multiple
    // pre-compiled objects in a single pre-compiled layer currently
    layer->SetPreCompiledObject(std::move(compiledBlobPtr));

    if (backend.has_value())
    {
        layer->SetBackendId(backend.value());
    }
    else if (layer->GetBackendHint().has_value())
    {
        layer->SetBackendId(layer->GetBackendHint().value());
    }

    return layer;
}

void NetworkImpl::ExecuteStrategy(IStrategy& strategy) const
{
    for (auto layer : GetGraph())
    {
        layer->ExecuteStrategy(strategy);
    };
}

OptimizedNetworkImpl::OptimizedNetworkImpl(const OptimizedNetworkImpl& other, const ModelOptions& modelOptions)
    : m_Graph(new Graph(*other.m_Graph.get()))
    , m_Guid(arm::pipe::IProfilingService::GetNextGuid())
    , m_ModelOptions(modelOptions)
{
}

OptimizedNetworkImpl::OptimizedNetworkImpl(std::unique_ptr<Graph> graph)
    : m_Graph(std::move(graph)), m_Guid(arm::pipe::IProfilingService::GetNextGuid())
{
}

OptimizedNetworkImpl::OptimizedNetworkImpl(std::unique_ptr<Graph> graph, const ModelOptions& modelOptions)
    : m_Graph(std::move(graph)), m_Guid(arm::pipe::IProfilingService::GetNextGuid()), m_ModelOptions(modelOptions)
{
}

OptimizedNetworkImpl::~OptimizedNetworkImpl()
{
}

void IOptimizedNetwork::ExecuteStrategy(IStrategy &strategy) const
{
    pOptimizedNetworkImpl->ExecuteStrategy(strategy);
}

void OptimizedNetworkImpl::ExecuteStrategy(IStrategy &strategy) const
{
    for (auto layer : GetGraph())
    {
        layer->ExecuteStrategy(strategy);
    };
}

} // namespace armnn
