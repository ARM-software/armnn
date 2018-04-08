//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "ClWorkloadFactory.hpp"

#include "armnn/Exceptions.hpp"
#include "armnn/Utils.hpp"

#include <string>
#include "CpuTensorHandle.hpp"
#include "Layer.hpp"
#include "Layers.hpp"

#ifdef ARMCOMPUTECL_ENABLED
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include "backends/MemCopyWorkload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "ClWorkloads.hpp"
#endif

#include "MakeWorkloadHelper.hpp"

#include <boost/polymorphic_cast.hpp>
#include <boost/format.hpp>

namespace armnn
{

bool ClWorkloadFactory::IsLayerSupported(const Layer& layer, DataType dataType, std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(Compute::GpuAcc, layer, dataType, outReasonIfUnsupported);
}

#ifdef ARMCOMPUTECL_ENABLED

ClWorkloadFactory::ClWorkloadFactory(IClTunedParameters* clTunedParameters):
    m_clTunedParameters(boost::polymorphic_downcast<ClTunedParameters*>(clTunedParameters))
{
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Select default platform as the first element
        cl::Platform::setDefault(platforms[0]);

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // Select default device as the first element
        cl::Device::setDefault(devices[0]);
    }
    catch (const cl::Error& clError)
    {
        throw ClRuntimeUnavailableException(boost::str(boost::format(
            "Could not initialize the CL runtime. Error description: %1%. CL error code: %2%"
        ) % clError.what() % clError.err()));
    }

    // Remove the use of global CL context
    cl::Context::setDefault(cl::Context{});
    BOOST_ASSERT(cl::Context::getDefault()() == NULL);

    // Remove the use of global CL command queue
    cl::CommandQueue::setDefault(cl::CommandQueue{});
    BOOST_ASSERT(cl::CommandQueue::getDefault()() == NULL);
}

ClWorkloadFactory::~ClWorkloadFactory()
{
}

void ClWorkloadFactory::LoadOpenClRuntime()
{
    cl::Device device = cl::Device::getDefault();
    cl::Context context;
    cl::CommandQueue commandQueue;

    try
    {
        arm_compute::CLKernelLibrary::get().clear_programs_cache();
        arm_compute::CLScheduler::get().init(context, commandQueue, device);
        arm_compute::CLKernelLibrary::get().init(".", context, device);

        context = cl::Context(device);

        bool enableProfiling = false;
#if ARMNN_PROFILING_ENABLED
        enableProfiling = true;
#endif
        if (m_clTunedParameters && m_clTunedParameters->m_Mode == IClTunedParameters::Mode::UpdateTunedParameters)
        {
            enableProfiling = true; // Needed for the CLTuner to work.
        }

        if (enableProfiling)
        {
            // Create a new queue with profiling enabled
            commandQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        }
        else
        {
            // Use default queue
            commandQueue = cl::CommandQueue(context, device);
        }
    }
    catch (const cl::Error& clError)
    {
        throw ClRuntimeUnavailableException(boost::str(boost::format(
            "Could not initialize the CL runtime. Error description: %1%. CL error code: %2%"
        ) % clError.what() % clError.err()));
    }

    // Note the first argument (path to cl source code) will be ignored as they should be embedded in the armcompute.
    arm_compute::CLKernelLibrary::get().init(".", context, device);

    arm_compute::ICLTuner* tuner = nullptr;
    if (m_clTunedParameters)
    {
        tuner = &m_clTunedParameters->m_Tuner;
    }
    arm_compute::CLScheduler::get().init(context, commandQueue, device, tuner);
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return std::make_unique<ClTensorHandle>(tensorInfo);
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateSubTensorHandle(ITensorHandle&      parent,
                                                                        TensorShape const&   subTensorShape,
                                                                        unsigned int const* subTensorOrigin) const
{
    BOOST_ASSERT(parent.GetType() == ITensorHandle::CL);

    arm_compute::Coordinates coords;
    arm_compute::TensorShape shape = armcomputetensorutils::BuildArmComputeTensorShape(subTensorShape);

    coords.set_num_dimensions(subTensorShape.GetNumDimensions());
    for (unsigned int i = 0; i < subTensorShape.GetNumDimensions(); i++)
    {
        // arm compute indexes tensor coords in reverse order
        unsigned int revertedIndex = subTensorShape.GetNumDimensions() - i - 1;
        coords.set(i, boost::numeric_cast<int>(subTensorOrigin[revertedIndex]));
    }

    return std::make_unique<ClSubTensorHandle>(static_cast<ClTensorHandle&>(parent).GetTensor(), shape, coords);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<CopyFromCpuToClFloat32Workload, CopyFromCpuToClUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<CopyFromClToCpuFloat32Workload, CopyFromClToCpuUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                               const WorkloadInfo&              info) const
{
    return MakeWorkload<ClActivationFloat32Workload, ClActivationUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                            const WorkloadInfo&           info) const
{
    return MakeWorkload<ClSoftmaxFloat32Workload, ClSoftmaxUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                             const WorkloadInfo&            info) const
{
    return MakeWorkload<ClSplitterFloat32Workload, ClSplitterUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                                  const WorkloadInfo&          info) const
{
    return MakeWorkload<ClMergerFloat32Workload, ClMergerUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClFullyConnectedFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&           info) const
{
    return MakeWorkload<ClPermuteFloat32Workload, ClPermuteUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&           info) const
{
    return MakeWorkload<ClPooling2dFloat32Workload, ClPooling2dUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                                         const WorkloadInfo&               info) const
{
    return MakeWorkload<ClConvolution2dFloat32Workload, ClConvolution2dUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClDepthwiseConvolutionFloat32Workload, ClDepthwiseConvolutionUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                                         const WorkloadInfo&                 info) const
{
    return MakeWorkload<ClNormalizationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&            info) const
{
    return MakeWorkload<ClAdditionFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClMultiplicationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClBatchNormalizationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    if (descriptor.m_Inputs.empty() || !descriptor.m_Inputs[0])
    {
        throw InvalidArgumentException("ClWorkloadFactory: Invalid null input for MemCopy workload");
    }

    // Create a workload that will copy tensor data from the inputs, which can have a number of different formats,
    // to CL tensors.
    switch (descriptor.m_Inputs[0]->GetType())
    {
    case ITensorHandle::Cpu:
        return MakeWorkload<CopyFromCpuToClFloat32Workload, CopyFromCpuToClUint8Workload>(descriptor, info);
#if ARMCOMPUTENEON_ENABLED
    case ITensorHandle::Neon:
    {
        return MakeWorkload<CopyFromNeonToClFloat32Workload, CopyFromNeonToClUint8Workload>(descriptor, info);
    }
#endif
    default:
        throw InvalidArgumentException("ClWorkloadFactory: Destination type not supported for MemCopy Workload.");
    }
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateResizeBilinear(
    const ResizeBilinearQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClResizeBilinearFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFakeQuantization(
    const FakeQuantizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClL2NormalizationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClConstantFloat32Workload, ClConstantUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClReshapeFloat32Workload, ClReshapeUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClFloorFloat32Workload, NullWorkload>(descriptor, info);
}

#else // #if ARMCOMPUTECL_ENABLED

ClWorkloadFactory::ClWorkloadFactory(IClTunedParameters* clTunedParameters)
{
    // No CL support
}

ClWorkloadFactory::~ClWorkloadFactory()
{
}

void ClWorkloadFactory::LoadOpenClRuntime()
{
    // No CL support
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return nullptr;
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateSubTensorHandle(ITensorHandle&      parent,
                                                                        TensorShape const&   subTensorShape,
                                                                        unsigned int const* subTensorOrigin) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                               const WorkloadInfo&              info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                            const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                             const WorkloadInfo&            info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                           const WorkloadInfo&          info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFullyConnected(const FullyConnectedQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&                  info) const
{
    return nullptr;
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                              const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                                  const WorkloadInfo&               info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                                  const WorkloadInfo&                 info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                             const WorkloadInfo&            info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMultiplication(const MultiplicationQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&                  info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateResizeBilinear(const ResizeBilinearQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFakeQuantization(const FakeQuantizationQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDetectionOutput(const DetectionOutputQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateReorg(const ReorgQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const
{
    return nullptr;
}

#endif // #if ARMCOMPUTECL_ENABLED

armnn::IClTunedParameters* IClTunedParameters::CreateRaw(armnn::IClTunedParameters::Mode mode)
{
    return new ClTunedParameters(mode);
}

armnn::IClTunedParametersPtr IClTunedParameters::Create(armnn::IClTunedParameters::Mode mode)
{
    return IClTunedParametersPtr(CreateRaw(mode), &IClTunedParameters::Destroy);
}

void IClTunedParameters::Destroy(IClTunedParameters* params)
{
    delete params;
}

ClTunedParameters::ClTunedParameters(armnn::IClTunedParameters::Mode mode)
    : m_Mode(mode)
#ifdef ARMCOMPUTECL_ENABLED
    , m_Tuner(mode == ClTunedParameters::Mode::UpdateTunedParameters)
#endif
{
}

void ClTunedParameters::Load(const char* filename)
{
#ifdef ARMCOMPUTECL_ENABLED
    try
    {
        m_Tuner.load_from_file(filename);
    }
    catch (const std::exception& e)
    {
        throw armnn::Exception(std::string("Failed to load tuned parameters file '") + filename + "': " +
            e.what());
    }
#endif
}

void ClTunedParameters::Save(const char* filename) const
{
#ifdef ARMCOMPUTECL_ENABLED
    try
    {
        m_Tuner.save_to_file(filename);
    }
    catch (const std::exception& e)
    {
        throw armnn::Exception(std::string("Failed to save tuned parameters file to '") + filename + "': " +
            e.what());
    }
#endif
}

} // namespace armnn
