//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "WorkloadFactory.hpp"
#include "OutputHandler.hpp"
#include "armnn/IRuntime.hpp"

#ifdef ARMCOMPUTECL_ENABLED
#include <arm_compute/runtime/CL/CLTuner.h>
#endif

namespace cl
{
class Context;
class CommandQueue;
class Device;
}

namespace armnn
{

class IClTunedParameters;

// ARM Compute OpenCL workload factory
class ClWorkloadFactory : public IWorkloadFactory
{
public:
    virtual ~ClWorkloadFactory(){};

    virtual Compute GetCompute() const override { return Compute::GpuAcc; }

    static bool IsLayerSupported(const Layer& layer, DataType dataType, std::string& outReasonIfUnsupported);

    void LoadOpenClRuntime(IClTunedParameters* clTunedParameters = nullptr);

    virtual bool SupportsSubTensors() const override { return true; }

    virtual std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle&      parent,
                                                                 TensorShape const&   subTensorShape,
                                                                 unsigned int const* subTensorOrigin) const override;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const override;

    virtual std::unique_ptr<IWorkload> CreateInput(const InputQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateOutput(const OutputQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const override;

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

    virtual std::unique_ptr<IWorkload> CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                      const WorkloadInfo&            info) const override;

    virtual std::unique_ptr<IWorkload> CreateMultiplication(const MultiplicationQueueDescriptor& descriptor,
                                                            const WorkloadInfo&                  info) const override;

    virtual std::unique_ptr<IWorkload> CreateBatchNormalization(const BatchNormalizationQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const override;

    virtual std::unique_ptr<IWorkload> CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info) const override;

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

    virtual std::unique_ptr<IWorkload> CreateFloor(const FloorQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const override;
};

class ClTunedParameters : public IClTunedParameters
{
public:
    ClTunedParameters(armnn::IClTunedParameters::Mode mode);

    virtual void Load(const char* filename);
    virtual void Save(const char* filename) const;

    Mode m_Mode;

#ifdef ARMCOMPUTECL_ENABLED
    arm_compute::CLTuner m_Tuner;
#endif
};

} // namespace armnn
