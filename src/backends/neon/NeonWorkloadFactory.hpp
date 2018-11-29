//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Optional.hpp>

#include <backendsCommon/OutputHandler.hpp>
#include <aclCommon/BaseMemoryManager.hpp>

#include <boost/core/ignore_unused.hpp>

namespace armnn
{

// Neon workload factory.
class NeonWorkloadFactory : public IWorkloadFactory
{
public:
    NeonWorkloadFactory(const std::shared_ptr<NeonMemoryManager>& memoryManager);

    const BackendId& GetBackendId() const override;

    static bool IsLayerSupported(const Layer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    virtual bool SupportsSubTensors() const override { return true; }

    virtual std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                                 TensorShape const& subTensorShape,
                                                                 unsigned int const* subTensorOrigin) const override;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const override;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                              DataLayout dataLayout) const override;

    virtual std::unique_ptr<IWorkload> CreateInput(const InputQueueDescriptor& descriptor,
                                                   const WorkloadInfo&        info) const override;

    virtual std::unique_ptr<IWorkload> CreateOutput(const OutputQueueDescriptor& descriptor,
                                                    const WorkloadInfo&        info) const override;

    virtual std::unique_ptr<IWorkload> CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                        const WorkloadInfo&              info) const override;

    virtual std::unique_ptr<IWorkload> CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                     const WorkloadInfo&           info) const override;

    virtual std::unique_ptr<IWorkload> CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                      const WorkloadInfo&            info) const override;

    virtual std::unique_ptr<IWorkload> CreateMerger(const MergerQueueDescriptor& descriptor,
                                                    const WorkloadInfo&          info) const override;

    virtual std::unique_ptr<IWorkload> CreateFullyConnected(const FullyConnectedQueueDescriptor& descriptor,
                                                            const WorkloadInfo&                  info) const override;

    virtual std::unique_ptr<IWorkload> CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                     const WorkloadInfo&           info) const override;

    virtual std::unique_ptr<IWorkload> CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                       const WorkloadInfo&           info) const override;

    virtual std::unique_ptr<IWorkload> CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                           const WorkloadInfo&               info) const override;

    virtual std::unique_ptr<IWorkload> CreateDepthwiseConvolution2d(
        const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                           const WorkloadInfo&                 info) const override;

    virtual std::unique_ptr<IWorkload> CreateMultiplication(const MultiplicationQueueDescriptor& descriptor,
                                                            const WorkloadInfo&                  info) const override;

    virtual std::unique_ptr<IWorkload> CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                      const WorkloadInfo&            info) const override;

    virtual std::unique_ptr<IWorkload> CreateBatchNormalization(const BatchNormalizationQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                     const WorkloadInfo&        info) const override;

    virtual std::unique_ptr<IWorkload> CreateResizeBilinear(const ResizeBilinearQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateFakeQuantization(const FakeQuantizationQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateConstant(const ConstantQueueDescriptor& descriptor,
                                                      const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateReshape(const ReshapeQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateSpaceToBatchNd(const SpaceToBatchNdQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateFloor(const FloorQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateLstm(const LstmQueueDescriptor& descriptor,
                                                  const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateConvertFp16ToFp32(const ConvertFp16ToFp32QueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateConvertFp32ToFp16(const ConvertFp32ToFp16QueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateDivision(const DivisionQueueDescriptor& descriptor,
                                                      const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateSubtraction(const SubtractionQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateMaximum(const MaximumQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateMean(const MeanQueueDescriptor& descriptor,
                                                  const WorkloadInfo& Info) const override;

    virtual std::unique_ptr<IWorkload> CreatePad(const PadQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateBatchToSpaceNd(const BatchToSpaceNdQueueDescriptor& descriptor,
                                                            const WorkloadInfo& Info) const override;

    virtual std::unique_ptr<IWorkload> CreateStridedSlice(const StridedSliceQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateMinimum(const MinimumQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info) const override;

private:
    mutable std::shared_ptr<NeonMemoryManager> m_MemoryManager;
};

} // namespace armnn
