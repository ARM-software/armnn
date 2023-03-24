//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/QuantizedLstmParams.hpp>
#include <armnn/TensorFwd.hpp>
#include <armnn/Types.hpp>

#include <Graph.hpp>
#include <Layer.hpp>
#include <OptimizedNetworkImpl.hpp>
#include <armnn/backends/SubgraphView.hpp>

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace armnn
{

class Graph;

using NetworkImplPtr = std::unique_ptr<NetworkImpl, void (*)(NetworkImpl* network)>;

/// Private implementation of INetwork.
class NetworkImpl
{
public:
    NetworkImpl(const NetworkOptions& networkOptions = {});
    ~NetworkImpl();

    const Graph& GetGraph() const
    { return *m_Graph; }

    Status PrintGraph();

    IConnectableLayer* AddInputLayer(LayerBindingId id, const char* name = nullptr);

    IConnectableLayer* AddActivationLayer(const ActivationDescriptor& activationDescriptor,
                                          const char* name = nullptr);
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddAdditionLayer(const char* name = nullptr);

    IConnectableLayer* AddArgMinMaxLayer(const ArgMinMaxDescriptor& desc,
                                         const char* name = nullptr);

    IConnectableLayer* AddBatchMatMulLayer(const BatchMatMulDescriptor& desc,
                                           const char* name = nullptr);

    IConnectableLayer* AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
                                                  const ConstTensor& mean,
                                                  const ConstTensor& variance,
                                                  const ConstTensor& beta,
                                                  const ConstTensor& gamma,
                                                  const char* name = nullptr);

    IConnectableLayer* AddBatchToSpaceNdLayer(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                              const char* name = nullptr);

    IConnectableLayer* AddCastLayer(const char* name = nullptr);

    IConnectableLayer* AddChannelShuffleLayer(const ChannelShuffleDescriptor& channelShuffleDescriptor,
                                              const char* name = nullptr);

    IConnectableLayer* AddComparisonLayer(const ComparisonDescriptor& comparisonDescriptor,
                                          const char* name = nullptr);

    IConnectableLayer* AddConcatLayer(const ConcatDescriptor& concatDescriptor,
                                      const char* name = nullptr);

    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const char* name = nullptr);

    IConnectableLayer* AddConvolution3dLayer(const Convolution3dDescriptor& convolution3dDescriptor,
                                             const char* name = nullptr);

    IConnectableLayer* AddConstantLayer(const ConstTensor& input, const char* name = nullptr);

    IConnectableLayer* AddDepthToSpaceLayer(const DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                            const char* name = nullptr);

    IConnectableLayer* AddDepthwiseConvolution2dLayer(const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
                                                      const char* name = nullptr);

    IConnectableLayer* AddDequantizeLayer(const char* name = nullptr);

    IConnectableLayer* AddDetectionPostProcessLayer(const DetectionPostProcessDescriptor& descriptor,
                                                    const ConstTensor& anchors,
                                                    const char* name = nullptr);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddDivisionLayer(const char* name = nullptr);

    IConnectableLayer* AddElementwiseBinaryLayer(const ElementwiseBinaryDescriptor& elementwiseBinaryDescriptor,
                                                 const char* name = nullptr);

    IConnectableLayer* AddElementwiseUnaryLayer(const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                                const char* name = nullptr);

    IConnectableLayer* AddMergeLayer(const char* name = nullptr);

    IConnectableLayer* AddFillLayer(const FillDescriptor& fillDescriptor,
                                    const char* name = nullptr);

    IConnectableLayer* AddFloorLayer(const char* name = nullptr);

    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const char* name = nullptr);

    IConnectableLayer* AddGatherLayer(const GatherDescriptor& gatherDescriptor,
                                      const char* name = nullptr);

    IConnectableLayer* AddGatherNdLayer(const char* name = nullptr);

    IConnectableLayer* AddInstanceNormalizationLayer(const InstanceNormalizationDescriptor& desc,
                                                     const char* name = nullptr);

    IConnectableLayer* AddL2NormalizationLayer(const L2NormalizationDescriptor& desc,
                                               const char* name = nullptr);

    IConnectableLayer* AddLogSoftmaxLayer(const LogSoftmaxDescriptor& logSoftmaxDescriptor,
                                          const char* name = nullptr);

    IConnectableLayer* AddLogicalBinaryLayer(const LogicalBinaryDescriptor& logicalBinaryDescriptor,
                                             const char* name = nullptr);

    IConnectableLayer* AddLstmLayer(const LstmDescriptor& descriptor,
                                    const LstmInputParams& params,
                                    const char* name = nullptr);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddMaximumLayer(const char* name = nullptr);

    IConnectableLayer* AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name = nullptr);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddMinimumLayer(const char* name = nullptr);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddMultiplicationLayer(const char* name = nullptr);

    IConnectableLayer* AddNormalizationLayer(const NormalizationDescriptor& normalizationDescriptor,
                                             const char* name = nullptr);

    IConnectableLayer* AddOutputLayer(LayerBindingId id, const char* name = nullptr);

    IConnectableLayer* AddPadLayer(const PadDescriptor& padDescriptor, const char* name = nullptr);

    IConnectableLayer* AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                       const char* name = nullptr);

    IConnectableLayer* AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
                                         const char* name = nullptr);

    IConnectableLayer* AddPooling3dLayer(const Pooling3dDescriptor& pooling3dDescriptor,
                                         const char* name = nullptr);

    IConnectableLayer* AddPrecompiledLayer(const PreCompiledDescriptor& preCompiledDescriptor,
                                           CompiledBlobPtr compiledBlobPtr,
                                           const Optional<BackendId>& backend,
                                           const char* name = nullptr);

    IConnectableLayer* AddPreluLayer(const char* name = nullptr);

    IConnectableLayer* AddQuantizeLayer(const char* name = nullptr);

    IConnectableLayer* AddQLstmLayer(const QLstmDescriptor& descriptor,
                                     const LstmInputParams& params,
                                     const char* name = nullptr);

    IConnectableLayer* AddQuantizedLstmLayer(const QuantizedLstmInputParams& params,
                                             const char* name = nullptr);

    IConnectableLayer* AddRankLayer(const char* name = nullptr);

    IConnectableLayer* AddReduceLayer(const ReduceDescriptor& reduceDescriptor,
                                      const char* name = nullptr);

    IConnectableLayer* AddResizeLayer(const ResizeDescriptor& resizeDescriptor,
                                      const char* name = nullptr);

    IConnectableLayer* AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                       const char* name = nullptr);

    IConnectableLayer* AddShapeLayer(const char* name = nullptr);

    IConnectableLayer* AddSliceLayer(const SliceDescriptor& sliceDescriptor, const char* name = nullptr);

    IConnectableLayer* AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
                                       const char* name = nullptr);

    IConnectableLayer* AddSplitterLayer(const ViewsDescriptor& splitterDescriptor,
                                        const char* name = nullptr);

    IConnectableLayer* AddSpaceToBatchNdLayer(const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                              const char* name = nullptr);

    IConnectableLayer* AddSpaceToDepthLayer(const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                            const char* name = nullptr);

    IConnectableLayer* AddStackLayer(const StackDescriptor& stackDescriptor,
                                     const char* name = nullptr);

    IConnectableLayer* AddStandInLayer(const StandInDescriptor& descriptor,
                                       const char* name = nullptr);

    IConnectableLayer* AddStridedSliceLayer(const StridedSliceDescriptor& stridedSliceDescriptor,
                                            const char* name = nullptr);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddSubtractionLayer(const char* name = nullptr);

    IConnectableLayer* AddSwitchLayer(const char* name = nullptr);

    IConnectableLayer* AddTransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& descriptor,
                                                      const ConstTensor& weights,
                                                      const Optional<ConstTensor>& biases,
                                                      const char* name = nullptr);

    IConnectableLayer* AddTransposeLayer(const TransposeDescriptor& transposeDescriptor,
                                         const char* name = nullptr);

    IConnectableLayer* AddUnidirectionalSequenceLstmLayer(const UnidirectionalSequenceLstmDescriptor& descriptor,
                                                          const LstmInputParams& params,
                                                          const char* name = nullptr);

    IConnectableLayer* AddConvertFp16ToFp32Layer(const char* name = nullptr);

    IConnectableLayer* AddConvertFp32ToFp16Layer(const char* name = nullptr);

    void ExecuteStrategy(IStrategy& strategy) const;

private:

    bool GetShapeInferenceMethod();
    bool GetAllowExpandedDims();
    NetworkOptions m_NetworkOptions;

    std::unique_ptr<Graph> m_Graph;
    ModelOptions           m_ModelOptions;
};

struct OptimizationResult
{
    bool m_Warning;
    bool m_Error;

    OptimizationResult(bool warning, bool error)
        : m_Warning(warning), m_Error(error)
    {}

    OptimizationResult()
        : OptimizationResult(false, false)
    {}

    bool IsOk() const
    { return !m_Warning && !m_Error; }
    bool IsWarningOnly() const
    { return m_Warning && !m_Error; }
    bool IsError() const
    { return m_Error; }

};

using BackendsMap = std::map<BackendId, std::unique_ptr<class IBackendInternal>>;

BackendsMap CreateSupportedBackends(TensorHandleFactoryRegistry& handleFactoryRegistry,
                                    struct BackendSettings& backendSettings);

OptimizationResult SelectTensorHandleStrategy(Graph& optGraph,
                                              BackendsMap& backends,
                                              TensorHandleFactoryRegistry& registry,
                                              bool importEnabled,
                                              bool exportEnabled,
                                              Optional<std::vector<std::string>&> errMessages);

OptimizationResult AssignBackends(OptimizedNetworkImpl* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  Graph::Iterator& firstLayer,
                                  Graph::Iterator& lastLayer,
                                  Optional<std::vector<std::string>&> errMessages);


OptimizationResult AssignBackends(OptimizedNetworkImpl* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  SubgraphView::IConnectableLayerIterator& firstLayer,
                                  SubgraphView::IConnectableLayerIterator& lastLayer,
                                  Optional<std::vector<std::string>&> errMessages);

struct OptimizerOptionsOpaqueImpl
{
    ~OptimizerOptionsOpaqueImpl() = default;

    explicit OptimizerOptionsOpaqueImpl()
            : m_ReduceFp32ToFp16(false)
            , m_Debug(false)
            , m_DebugToFile(false)
            , m_ReduceFp32ToBf16(false)
            , m_shapeInferenceMethod(armnn::ShapeInferenceMethod::ValidateOnly)
            , m_ImportEnabled(false)
            , m_ModelOptions()
            , m_ProfilingEnabled(false)
            , m_ExportEnabled(false)
            , m_AllowExpandedDims(false)
    {
    }

    explicit OptimizerOptionsOpaqueImpl(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16,
                                        bool importEnabled, ModelOptions modelOptions = {},
                                        bool exportEnabled = false, bool debugToFile = false)
            : m_ReduceFp32ToFp16(reduceFp32ToFp16)
            , m_Debug(debug)
            , m_DebugToFile(debugToFile)
            , m_ReduceFp32ToBf16(reduceFp32ToBf16)
            , m_shapeInferenceMethod(armnn::ShapeInferenceMethod::ValidateOnly)
            , m_ImportEnabled(importEnabled)
            , m_ModelOptions(modelOptions)
            , m_ProfilingEnabled(false)
            , m_ExportEnabled(exportEnabled)
            , m_AllowExpandedDims(false)
    {
    }

    explicit OptimizerOptionsOpaqueImpl(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16,
                                        ShapeInferenceMethod shapeInferenceMethod,
                                        bool importEnabled, ModelOptions modelOptions, bool exportEnabled,
                                        bool debugToFile, bool allowExpandedDims)
            : m_ReduceFp32ToFp16(reduceFp32ToFp16)
            , m_Debug(debug)
            , m_DebugToFile(debugToFile)
            , m_ReduceFp32ToBf16(reduceFp32ToBf16)
            , m_shapeInferenceMethod(shapeInferenceMethod)
            , m_ImportEnabled(importEnabled)
            , m_ModelOptions(modelOptions)
            , m_ProfilingEnabled(false)
            , m_ExportEnabled(exportEnabled)
            , m_AllowExpandedDims(allowExpandedDims)
    {
    }

    /// Reduces all Fp32 operators in the model to Fp16 for faster processing.
    /// @Note This feature works best if all operators of the model are in Fp32. ArmNN will add conversion layers
    ///       between layers that weren't in Fp32 in the first place or if the operator is not supported in Fp16.
    ///       The overhead of these conversions can lead to a slower overall performance if too many conversions are
    ///       required.
    bool m_ReduceFp32ToFp16 = false;

    /// Add debug data for easier troubleshooting
    bool m_Debug = false;

    /// Pass debug data to separate output files for easier troubleshooting
    bool m_DebugToFile = false;

    /// @Note This feature has been replaced by enabling Fast Math in compute library backend options.
    /// This is currently a placeholder option
    bool m_ReduceFp32ToBf16 = false;

    /// Infer output size when not available
    ShapeInferenceMethod m_shapeInferenceMethod = armnn::ShapeInferenceMethod::ValidateOnly;

    /// Enable Import
    bool m_ImportEnabled = false;

    /// Enable Model Options
    ModelOptions m_ModelOptions;

    /// Enable profiling dump of the optimizer phase
    bool m_ProfilingEnabled = false;

    /// Enable Export
    bool m_ExportEnabled = false;

    /// When calculating tensor sizes, dimensions of size == 1 will be ignored
    bool m_AllowExpandedDims = false;
};

} // namespace armnn
