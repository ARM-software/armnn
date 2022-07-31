//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/IRuntime.hpp>
#include <armnn/Optional.hpp>

#include <armnn/backends/IBackendInternal.hpp>

#include <backendsCommon/WorkloadFactoryBase.hpp>
#include <aclCommon/BaseMemoryManager.hpp>

#include <arm_compute/core/CL/CLCompileContext.h>

namespace armnn
{

// ARM Compute OpenCL workload factory.
class ClWorkloadFactory : public WorkloadFactoryBase
{
public:
    ClWorkloadFactory(const std::shared_ptr<ClMemoryManager>& memoryManager);

    ClWorkloadFactory(const std::shared_ptr<ClMemoryManager>& memoryManager,
                      const IBackendInternal::IBackendSpecificModelContextPtr& modelContextPtr);

    void AfterWorkloadsCreated() override;

    const BackendId& GetBackendId() const override;

    static bool IsLayerSupported(const Layer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    static bool IsLayerSupported(const IConnectableLayer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported,
                                 const ModelOptions& modelOptions);

    bool SupportsSubTensors() const override { return true; }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateSubTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override;

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      const bool IsMemoryManaged = true) const override;

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
                                                      const bool IsMemoryManaged = true) const override;

    std::unique_ptr<IWorkload> CreateWorkload(LayerType type,
                                              const QueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateAddition(const AdditionQueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateArgMinMax(const ArgMinMaxQueueDescriptor& descriptor,
                                               const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateBatchNormalization(const BatchNormalizationQueueDescriptor& descriptor,
                                                        const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateBatchToSpaceNd(const BatchToSpaceNdQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateCast(const CastQueueDescriptor& descriptor,
                                          const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateChannelShuffle(const ChannelShuffleQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateComparison(const ComparisonQueueDescriptor& descriptor,
                                                const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateConcat(const ConcatQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateConstant(const ConstantQueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateConvertFp16ToFp32(const ConvertFp16ToFp32QueueDescriptor& descriptor,
                                                       const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateConvertFp32ToFp16(const ConvertFp32ToFp16QueueDescriptor& descriptor,
                                                       const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateConvolution3d(const Convolution3dQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateDebug(const DebugQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateDepthToSpace(const DepthToSpaceQueueDescriptor& descriptor,
                                                  const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateDepthwiseConvolution2d(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateDequantize(const DequantizeQueueDescriptor& descriptor,
                                                const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateDetectionPostProcess(const DetectionPostProcessQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateDivision(const DivisionQueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateElementwiseUnary(const ElementwiseUnaryQueueDescriptor& descriptor,
                                                      const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateFill(const FillQueueDescriptor& descriptor,
                                          const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateFloor(const FloorQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateFullyConnected(const FullyConnectedQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateGather(const GatherQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateInput(const InputQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateInstanceNormalization(const InstanceNormalizationQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateLogicalBinary(const LogicalBinaryQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateLogSoftmax(const LogSoftmaxQueueDescriptor& descriptor,
                                                const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateLstm(const LstmQueueDescriptor& descriptor,
                                          const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateMaximum(const MaximumQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateMean(const MeanQueueDescriptor& descriptor,
                                          const WorkloadInfo& Info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateMemImport(const MemImportQueueDescriptor& descriptor,
                                               const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateMinimum(const MinimumQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateMultiplication(const MultiplicationQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateOutput(const OutputQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreatePad(const PadQueueDescriptor& descriptor,
                                         const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreatePermute(const PermuteQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                               const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreatePrelu(const PreluQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateQLstm(const QLstmQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateQuantize(const QuantizeQueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateQuantizedLstm(const QuantizedLstmQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateRank(const RankQueueDescriptor& descriptor,
                                          const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateReduce(const ReduceQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateReshape(const ReshapeQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateResize(const ResizeQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateSlice(const SliceQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateSpaceToBatchNd(const SpaceToBatchNdQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateSpaceToDepth(const SpaceToDepthQueueDescriptor& descriptor,
                                                  const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateStack(const StackQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateStridedSlice(const StridedSliceQueueDescriptor& descriptor,
                                                  const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateSubtraction(const SubtractionQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateTranspose(const TransposeQueueDescriptor& descriptor,
                                               const WorkloadInfo& info) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable "
    "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.", "23.08")
    std::unique_ptr<IWorkload> CreateTransposeConvolution2d(const TransposeConvolution2dQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const override;

private:
    template<typename FloatWorkload, typename Uint8Workload, typename QueueDescriptorType, typename... Args>
    static std::unique_ptr<IWorkload> MakeWorkload(const QueueDescriptorType& descriptor,
                                                   const WorkloadInfo& info,
                                                   Args&&... args);

    template <typename Workload, typename QueueDescriptorType, typename... Args>
    static std::unique_ptr<IWorkload> MakeWorkload(const QueueDescriptorType& descriptor,
                                                   const WorkloadInfo& info,
                                                   Args&&... args);

    void InitializeCLCompileContext();

    mutable std::shared_ptr<ClMemoryManager> m_MemoryManager;
    const IBackendInternal::IBackendSpecificModelContextPtr m_ModelContextPtr;
    arm_compute::CLCompileContext m_CLCompileContext;
};

} // namespace armnn
