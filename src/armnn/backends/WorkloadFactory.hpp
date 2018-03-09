//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "Workload.hpp"
#include <memory>
#include "armnn/TensorFwd.hpp"
#include "OutputHandler.hpp"

namespace armnn
{

class Layer;

// Workload factory interface for compute backends
class IWorkloadFactory
{
public:
    virtual ~IWorkloadFactory() { }

    virtual Compute GetCompute() const = 0;

    static bool IsLayerSupported(Compute compute, const Layer& layer, DataType dataType,
        std::string& outReasonIfUnsupported);
    static bool IsLayerSupported(const Layer& layer, DataType dataType, std::string& outReasonIfUnsupported);

    virtual bool SupportsSubTensors() const = 0;

    virtual std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                                 TensorShape const& subTensorShape,
                                                                 unsigned int const* subTensorOrigin
                                                                ) const = 0;

    virtual std::unique_ptr<IWorkload> CreateInput(const InputQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const = 0;

    virtual std::unique_ptr<IWorkload> CreateOutput(const OutputQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                        const WorkloadInfo&              info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                     const WorkloadInfo&           info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                      const WorkloadInfo&            info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateMerger(const MergerQueueDescriptor& descriptor,
                                                    const WorkloadInfo&          info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateFullyConnected(const FullyConnectedQueueDescriptor& descriptor,
                                                            const WorkloadInfo&                  info) const = 0;

    virtual std::unique_ptr<IWorkload> CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                     const WorkloadInfo&           info) const = 0;

    virtual std::unique_ptr<IWorkload> CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                       const WorkloadInfo&           info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                           const WorkloadInfo&               info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateDepthwiseConvolution2d(
        const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                           const WorkloadInfo&                 info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                      const WorkloadInfo&            info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateMultiplication(const MultiplicationQueueDescriptor& descriptor,
                                                            const WorkloadInfo&                  info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateBatchNormalization(const BatchNormalizationQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateResizeBilinear(const ResizeBilinearQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateFakeQuantization(const FakeQuantizationQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateConstant(const ConstantQueueDescriptor& descriptor,
                                                      const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateReshape(const ReshapeQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info) const = 0;

    virtual std::unique_ptr<IWorkload> CreateFloor(const FloorQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const = 0;
};

} //namespace armnn