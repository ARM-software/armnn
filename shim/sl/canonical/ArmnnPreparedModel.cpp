//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "arm-armnn-sl"

#include "ArmnnPreparedModel.hpp"
#include "CanonicalUtils.hpp"

#include <DefaultExecution.h>
#include <LegacyUtils.h>
#include <nnapi/IBurst.h>
#include <nnapi/IPreparedModel.h>
#include <nnapi/Result.h>
#include <nnapi/SharedMemory.h>
#include <nnapi/TypeUtils.h>
#include <nnapi/Types.h>
#include <nnapi/Validation.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

using namespace android;
using namespace android::nn;

static const Timing g_NoTiming = {};

namespace {

using namespace armnn_driver;

unsigned long MicrosecondsDuration(android::nn::TimePoint endPoint, android::nn::TimePoint startPoint)
{
    return static_cast<unsigned long>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      endPoint - startPoint).count());
}

bool ValidateRequestArgument(const Request::Argument& requestArg, const armnn::TensorInfo& tensorInfo)
{
    if (requestArg.dimensions.size() != 0)
    {
        if (requestArg.dimensions.size() != tensorInfo.GetNumDimensions())
        {
            VLOG(DRIVER) << "Mismatched dimensions (request argument: "
                         << requestArg.dimensions.size() << " expected: " << tensorInfo.GetNumDimensions();
            return false;
        }

        for (unsigned int d = 0; d < tensorInfo.GetNumDimensions(); ++d)
        {
            if (requestArg.dimensions[d] != 0 && requestArg.dimensions[d] != tensorInfo.GetShape()[d])
            {
                VLOG(DRIVER) << "Mismatched dimensions " << d
                             << " (request argument: " << requestArg.dimensions[d]
                             << " expected: " << tensorInfo.GetShape()[d];
                return false;
            }
        }
    }

    return true;
}

armnn::Tensor GetTensorForRequestArgument(const Request::Argument& requestArg,
                                          const armnn::TensorInfo& tensorInfo,
                                          const std::vector<::android::nn::RunTimePoolInfo>& requestPools)
{
    if (!ValidateRequestArgument(requestArg, tensorInfo))
    {
        return armnn::Tensor();
    }

    if (requestArg.lifetime == Request::Argument::LifeTime::POINTER)
    {
        return armnn::Tensor(tensorInfo, GetMemoryFromPointer(requestArg));
    }
    else if (requestArg.lifetime == Request::Argument::LifeTime::POOL)
    {
        return armnn::Tensor(tensorInfo, GetMemoryFromPool(requestArg.location, requestPools));
    }
    return armnn::Tensor();
}

inline std::string BuildTensorName(const char* tensorNamePrefix, std::size_t index)
{
    return tensorNamePrefix + std::to_string(index);
}

bool IsPointerTypeMemory(const Request& request)
{
    for (auto& input : request.inputs)
    {
        if (input.lifetime != Request::Argument::LifeTime::POINTER)
        {
            return false;
        }
    }

    for (auto& output: request.outputs)
    {
        if (output.lifetime != Request::Argument::LifeTime::POINTER)
        {
           return false;
        }
    }

    return true;
}

} // anonymous namespace

using namespace android::nn;

namespace armnn_driver
{

void ArmnnPreparedModel::Init()
{
    // Enable profiling if required.
    m_Runtime->GetProfiler(m_NetworkId)->EnableProfiling(m_GpuProfilingEnabled);
}

ArmnnPreparedModel::ArmnnPreparedModel(armnn::NetworkId networkId,
                                       armnn::IRuntime* runtime,
                                       const Model& model,
                                       const std::string& requestInputsAndOutputsDumpDir,
                                       const bool gpuProfilingEnabled,
                                       Priority priority)
    : m_NetworkId(networkId)
    , m_Runtime(runtime)
    , m_Model(model)
    , m_RequestInputsAndOutputsDumpDir(requestInputsAndOutputsDumpDir)
    , m_GpuProfilingEnabled(gpuProfilingEnabled)
    , m_ModelPriority(priority)
    , m_PrepareFromCache(false)
{
    Init();
}

ArmnnPreparedModel::ArmnnPreparedModel(armnn::NetworkId networkId,
                                       armnn::IRuntime* runtime,
                                       const std::string& requestInputsAndOutputsDumpDir,
                                       const bool gpuProfilingEnabled,
                                       Priority priority,
                                       const bool prepareModelFromCache)
    : m_NetworkId(networkId)
    , m_Runtime(runtime)
    , m_RequestInputsAndOutputsDumpDir(requestInputsAndOutputsDumpDir)
    , m_GpuProfilingEnabled(gpuProfilingEnabled)
    , m_ModelPriority(priority)
    , m_PrepareFromCache(prepareModelFromCache)
{
    Init();
}


ErrorStatus ArmnnPreparedModel::PrepareMemoryForInputs(
    armnn::InputTensors& inputs,
    const Request& request,
    const std::vector<android::nn::RunTimePoolInfo>& memPools) const
{
    inputs.reserve(request.inputs.size());
    for (unsigned int i = 0; i < request.inputs.size(); i++)
    {
        const auto& inputArg = request.inputs[i];

        armnn::TensorInfo inputTensorInfo = m_Runtime->GetInputTensorInfo(m_NetworkId, i);
        // inputs (of type InputTensors) is composed of a vector of ConstTensors.
        // Therefore, set all TensorInfo isConstant parameters of input Tensors to true.
        inputTensorInfo.SetConstant();
        const armnn::Tensor inputTensor = GetTensorForRequestArgument(inputArg, inputTensorInfo, memPools);

        if (inputTensor.GetMemoryArea() == nullptr)
        {
            VLOG(DRIVER) << "Cannot execute request. Error converting request input " << i << "to tensor.";
            return ErrorStatus::GENERAL_FAILURE;
        }
        inputs.emplace_back(i, inputTensor);
    }

    return ErrorStatus::NONE;
}

ErrorStatus ArmnnPreparedModel::PrepareMemoryForOutputs(
    armnn::OutputTensors& outputs,
    std::vector<OutputShape> &outputShapes,
    const Request& request,
    const std::vector<android::nn::RunTimePoolInfo>& memPools) const
{
    outputs.reserve(request.outputs.size());
    for (unsigned int i = 0; i < request.outputs.size(); i++)
    {
        auto& outputArg = request.outputs[i];

        armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, i);
        armnn::Tensor outputTensor = GetTensorForRequestArgument(outputArg, outputTensorInfo, memPools);
        if (outputTensor.GetMemoryArea() == nullptr)
        {
            VLOG(DRIVER) << "Cannot execute request. Error converting request output " << i << "to tensor.";
            return ErrorStatus::GENERAL_FAILURE;
        }

        const size_t outputSize = outputTensorInfo.GetNumBytes();

        unsigned int count = 0;
        std::for_each(outputArg.dimensions.begin(), outputArg.dimensions.end(), [&](auto dim)
        {
            if (dim != 0)
            {
                outputTensorInfo.GetShape()[count] = dim;
            }
            else
            {
                outputTensorInfo.GetShape()[count] = outputArg.dimensions.size();
            }

            count++;
        });

        outputs.emplace_back(i, outputTensor);
        outputShapes[i] = ComputeShape(outputTensorInfo);

        if (outputArg.location.length < outputSize)
        {
            VLOG(DRIVER) << "ArmnnPreparedModel::Execute failed outputArg.location.length "
                  << std::to_string(outputArg.location.length).c_str()
                  << " < outputSize " << std::to_string(outputSize).c_str();
            outputShapes[i].isSufficient = false;
            return ErrorStatus::OUTPUT_INSUFFICIENT_SIZE;
        }

        //TODO: Need to check for Request::Argument::LifeTime::POINTER
        if (outputArg.lifetime == Request::Argument::LifeTime::POOL)
        {
            size_t bufferSize = memPools.at(outputArg.location.poolIndex).getSize();
            if (bufferSize < outputSize)
            {
                VLOG(DRIVER) << "ArmnnPreparedModel::Execute failed bufferSize "
                             << std::to_string(outputArg.location.length).c_str()
                             << " < outputSize " << std::to_string(outputSize).c_str();
                outputShapes[i].isSufficient = false;
                return ErrorStatus::OUTPUT_INSUFFICIENT_SIZE;
            }
        }
    }
    return ErrorStatus::NONE;
}

ErrorStatus ArmnnPreparedModel::PrepareMemoryForIO(armnn::InputTensors& inputs,
                                                   armnn::OutputTensors& outputs,
                                                   std::vector<android::nn::RunTimePoolInfo>& memPools,
                                                   const Request& request,
                                                   const bool pointerMemory) const
{
    //Check memory pools are not empty
    // add the inputs and outputs with their data
    try
    {
        if (!pointerMemory && !setRunTimePoolInfosFromMemoryPools(&memPools, request.pools))
        {
            return ErrorStatus::INVALID_ARGUMENT;
        }

        if (PrepareMemoryForInputs(inputs, request, memPools) != ErrorStatus::NONE)
        {
            VLOG(DRIVER) << "Failed when preparing memory for Inputs";
            return ErrorStatus::GENERAL_FAILURE;
        }

        std::vector<OutputShape> outputShapes(request.outputs.size());

        auto errorStatus = PrepareMemoryForOutputs(outputs, outputShapes, request, memPools);
        if (errorStatus != ErrorStatus::NONE)
        {
            return errorStatus;
        }
    }
    catch (armnn::Exception& e)
    {
        VLOG(DRIVER) << "armnn::Exception caught while preparing for EnqueueWorkload: " << e.what();
        return ErrorStatus::GENERAL_FAILURE;
    }
    catch (std::exception& e)
    {
        VLOG(DRIVER) << "std::exception caught while preparing for EnqueueWorkload: " << e.what();
        return ErrorStatus::GENERAL_FAILURE;
    }

    return ErrorStatus::NONE;
}

ExecutionResult<std::pair<std::vector<OutputShape>, Timing>> ArmnnPreparedModel::execute(
    const Request& request,
    MeasureTiming measureTiming,
    const OptionalTimePoint& deadline,
    const OptionalDuration&,
    const std::vector<android::nn::TokenValuePair>& hints,
    const std::vector<android::nn::ExtensionNameAndPrefix>& extensionNameToPrefix) const
{
    VLOG(DRIVER) << "CanonicalDriver::PreparedModel::execute()";

    CanonicalExecutionContext ctx;
    if (measureTiming == MeasureTiming::YES)
    {
        ctx.measureTimings = measureTiming;
        ctx.driverStart =  Clock::now();
    }

    if (!m_PrepareFromCache)
    {
        const auto modelRequest = validateRequestForModel(request, m_Model);
        if (!modelRequest.ok())
        {
            return NN_ERROR(ErrorStatus::INVALID_ARGUMENT) << modelRequest.error();
        }
        VLOG(DRIVER) << "ArmnnPreparedModel::execute(): " << GetModelSummary(m_Model).c_str();
    }
    if (hasDeadlinePassed(deadline))
    {
        return NN_ERROR(ErrorStatus::MISSED_DEADLINE_PERSISTENT);
    }

    // map the memory pool into shared pointers
    // use a shared memory pools vector on the heap, as it is passed to the request thread
    auto memPools = std::make_shared<std::vector<android::nn::RunTimePoolInfo>>();

    // allocate the tensors on the heap, as they are passed to the request thread
    auto inputTensors = std::make_shared<armnn::InputTensors>();
    auto outputTensors = std::make_shared<armnn::OutputTensors>();

    auto isPointerTypeMemory = IsPointerTypeMemory(request);
    ErrorStatus theErrorStatus = PrepareMemoryForIO(*inputTensors,
                                                    *outputTensors,
                                                    *memPools,
                                                    request,
                                                    isPointerTypeMemory);

    switch(theErrorStatus)
    {
        case ErrorStatus::OUTPUT_INSUFFICIENT_SIZE:
            return NN_ERROR(ErrorStatus::OUTPUT_INSUFFICIENT_SIZE);
        case ErrorStatus::GENERAL_FAILURE:
            return NN_ERROR(ErrorStatus::GENERAL_FAILURE);
        case ErrorStatus::INVALID_ARGUMENT:
            return NN_ERROR(ErrorStatus::INVALID_ARGUMENT);
        default:
        {}
    }

    std::vector<OutputShape> outputShapes(outputTensors->size());
    for (unsigned int i = 0; i < outputTensors->size(); i++)
    {
        std::pair<int, armnn::Tensor> outputTensorPair = (*outputTensors)[i];
        const armnn::Tensor outputTensor = outputTensorPair.second;
        const armnn::TensorInfo outputTensorInfo = outputTensor.GetInfo();

        outputShapes[i] = ComputeShape(outputTensorInfo);
    }
    Timing theTiming;

    VLOG(DRIVER) << "ArmnnPreparedModel::execute(...) before ExecuteGraph";
    auto errorStatus = ExecuteGraph(memPools, *inputTensors, *outputTensors, ctx, isPointerTypeMemory);
    if (errorStatus != ErrorStatus::NONE)
    {
        return NN_ERROR(errorStatus) << "execute() failed";
    }
    VLOG(DRIVER) << "ArmnnPreparedModel::execute(...) after ExecuteGraph";

    return std::make_pair(outputShapes, theTiming);
}

ErrorStatus ArmnnPreparedModel::ExecuteGraph(
    std::shared_ptr<std::vector<android::nn::RunTimePoolInfo>>& pMemPools,
    armnn::InputTensors& inputTensors,
    armnn::OutputTensors& outputTensors,
    CanonicalExecutionContext ctx,
    const bool pointerMemory) const
{
    VLOG(DRIVER) << "ArmnnPreparedModel::ExecuteGraph(...)";

    DumpTensorsIfRequired("Input", inputTensors);
    std::vector<armnn::ImportedInputId> importedInputIds;
    std::vector<armnn::ImportedOutputId> importedOutputIds;
    try
    {
        if (ctx.measureTimings == MeasureTiming::YES)
        {
            ctx.deviceStart =  Clock::now();
        }
        armnn::Status status;
        VLOG(DRIVER) << "ArmnnPreparedModel::ExecuteGraph m_AsyncModelExecutionEnabled false";
        importedInputIds = m_Runtime->ImportInputs(m_NetworkId, inputTensors, armnn::MemorySource::Malloc);
        if (!importedInputIds.empty())
        {
            // Some or all of the input tensors been imported. We need to remove the ones that could from
            // inputTensors.
            for (armnn::ImportedInputId& importedId : importedInputIds)
            {
                inputTensors.erase(
                        std::remove_if(
                                inputTensors.begin(), inputTensors.end(),
                                [&importedId](std::pair<armnn::LayerBindingId, class armnn::ConstTensor>& element) {
                                    return (element.first == static_cast<int>(importedId));
                                }),
                        inputTensors.end());
            }
        }
        importedOutputIds = m_Runtime->ImportOutputs(m_NetworkId, outputTensors, armnn::MemorySource::Malloc);
        if (!importedOutputIds.empty())
        {
            // Some or all of the output tensors could not be imported. We need to remove the ones that could
            // from outputTensors.
            for (armnn::ImportedInputId& importedId : importedOutputIds)
            {
                outputTensors.erase(
                        std::remove_if(
                                outputTensors.begin(), outputTensors.end(),
                                [&importedId](std::pair<armnn::LayerBindingId, class armnn::Tensor>& element) {
                                    return (element.first == static_cast<int>(importedId));
                                }),
                        outputTensors.end());
            }
        }
        status = m_Runtime->EnqueueWorkload(m_NetworkId,
                                            inputTensors,
                                            outputTensors,
                                            importedInputIds,
                                            importedOutputIds);

        if (ctx.measureTimings == MeasureTiming::YES)
        {
            ctx.deviceEnd =  Clock::now();
        }
        if (status != armnn::Status::Success)
        {
            VLOG(DRIVER) << "ArmnnPreparedModel:ExecuteGraph EnqueueWorkload failed";
            return ErrorStatus::GENERAL_FAILURE;
        }
    }
    catch (armnn::Exception& e)
    {
        VLOG(DRIVER) << "armnn:Exception caught from EnqueueWorkload: " << e.what();
        return ErrorStatus::GENERAL_FAILURE;
    }
    catch (std::exception& e)
    {
        VLOG(DRIVER) << "std::exception caught from EnqueueWorkload: " << e.what();
        return ErrorStatus::GENERAL_FAILURE;
    }

    if (!pointerMemory && (!importedInputIds.empty() || !importedOutputIds.empty()))
    {
        CommitPools(*pMemPools);
    }
    DumpTensorsIfRequired("Output", outputTensors);

    if (ctx.measureTimings == MeasureTiming::YES)
    {
        ctx.driverEnd =  Clock::now();
        Timing timing;
        timing.timeOnDevice = ctx.deviceEnd - ctx.deviceStart;
        timing.timeInDriver = ctx.driverEnd - ctx.driverStart;
        VLOG(DRIVER) << "ArmnnPreparedModel::execute timing - Device = "
                     << timing.timeOnDevice << "Driver = " <<  timing.timeInDriver;
    }
    return ErrorStatus::NONE;
}

Priority ArmnnPreparedModel::GetModelPriority() const
{
    return m_ModelPriority;
}


GeneralResult<std::pair<SyncFence, ExecuteFencedInfoCallback>> ArmnnPreparedModel::executeFenced(
    const Request& request,
    const std::vector<SyncFence>& waitFor,
    MeasureTiming measureTiming,
    const OptionalTimePoint& deadline,
    const OptionalDuration&,
    const OptionalDuration&,
    const std::vector<android::nn::TokenValuePair>& hints,
    const std::vector<android::nn::ExtensionNameAndPrefix>& extensionNameToPrefix) const
{
    VLOG(DRIVER) << "ArmnnPreparedModel::executeFenced()";

    if (!m_PrepareFromCache) {
        const auto modelRequest = validateRequestForModel(request, m_Model);
        if (!modelRequest.ok())
        {
            return NN_ERROR(ErrorStatus::INVALID_ARGUMENT) << modelRequest.error();
        }
        VLOG(DRIVER) << "ArmnnPreparedModel::executeFenced(): " << GetModelSummary(m_Model).c_str();
    }
    if (hasDeadlinePassed(deadline))
    {
        return NN_ERROR(ErrorStatus::MISSED_DEADLINE_PERSISTENT);
    }

    CanonicalExecutionContext ctx;
    if (measureTiming == MeasureTiming::YES)
    {
        ctx.measureTimings = measureTiming;
        ctx.driverStart =  Clock::now();
    }

    // Wait for the dependent events to signal
    for (const auto& syncFence : waitFor)
    {
        if (!syncFence.getSharedHandle())
        {
            return NN_ERROR(ErrorStatus::INVALID_ARGUMENT);
        }
        if (syncFence.syncWait({}) != SyncFence::FenceState::SIGNALED)
        {
            return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << "syncWait failed";
        }
    }

    android::nn::TimePoint fenceExecutionStart;
    if (measureTiming == MeasureTiming::YES)
    {
        fenceExecutionStart = Clock::now();
    }

    // map the memory pool into shared pointers
    // use a shared memory pools vector on the heap, as it is passed to the request thread
    auto memPools = std::make_shared<std::vector<android::nn::RunTimePoolInfo>>();

    // allocate the tensors on the heap, as they are passed to the request thread
    auto inputTensors = std::make_shared<armnn::InputTensors>();
    auto outputTensors = std::make_shared<armnn::OutputTensors>();

    auto isPointerTypeMemory = IsPointerTypeMemory(request);
    ErrorStatus theErrorStatus = PrepareMemoryForIO(*inputTensors,
                                                    *outputTensors,
                                                    *memPools,
                                                    request,
                                                    isPointerTypeMemory);

    if (theErrorStatus != ErrorStatus::NONE)
    {
        return NN_ERROR(ErrorStatus::INVALID_ARGUMENT) << "executeFenced() failed";
    }

    Timing timingSinceLaunch = {};
    Timing timingAfterFence  = {};
    if (measureTiming == MeasureTiming::YES)
    {
        timingAfterFence.timeOnDevice = ctx.deviceEnd - ctx.deviceStart;
        timingAfterFence.timeInDriver = ctx.driverEnd - fenceExecutionStart;
        VLOG(DRIVER) << "executeFenced timingSinceLaunch = " << timingAfterFence.timeOnDevice;
        VLOG(DRIVER) << "executeFenced timingAfterFence = " << timingAfterFence.timeInDriver;
    }

    VLOG(DRIVER) << "ArmnnCanonicalPreparedModel::executeFenced(...) before ExecuteGraph";
    auto errorStatus = ExecuteGraph(memPools, *inputTensors, *outputTensors, ctx, isPointerTypeMemory);
    VLOG(DRIVER) << "ArmnnCanonicalPreparedModel::executeFenced(...) after ExecuteGraph";

    ExecuteFencedInfoCallback armnnFencedExecutionCallback =
            [timingSinceLaunch, timingAfterFence, errorStatus]() {

                GeneralResult<std::pair<Timing, Timing>> result;

                switch(errorStatus)
                {
                    case ErrorStatus::OUTPUT_INSUFFICIENT_SIZE:
                        result.error().code = (ErrorStatus::OUTPUT_INSUFFICIENT_SIZE);
                    case ErrorStatus::GENERAL_FAILURE:
                        result.error().code = (ErrorStatus::GENERAL_FAILURE);
                    case ErrorStatus::INVALID_ARGUMENT:
                        result.error().code = (ErrorStatus::INVALID_ARGUMENT);
                    default:
                    {
                        result.value() = std::make_pair(timingSinceLaunch, timingAfterFence);
                    }
                }
                return result;
            };
    return std::make_pair(SyncFence::createAsSignaled(), std::move(armnnFencedExecutionCallback ));
}

GeneralResult<SharedExecution> ArmnnPreparedModel::createReusableExecution(
    const Request& request,
    MeasureTiming measureTiming,
    const OptionalDuration& loopTimeoutDuration,
    const std::vector<android::nn::TokenValuePair>& hints,
    const std::vector<android::nn::ExtensionNameAndPrefix>& extensionNameToPrefix) const
{
    VLOG(DRIVER) << "ArmnnPreparedModel::createReusableExecution()";
    return std::make_shared<DefaultExecution>(shared_from_this(),
                                              request,
                                              measureTiming,
                                              loopTimeoutDuration);
}

GeneralResult<SharedBurst> ArmnnPreparedModel::configureExecutionBurst() const
{
    // TODO: Implement BURST
    return nullptr;
}

std::any ArmnnPreparedModel::getUnderlyingResource() const
{
    return &m_Model;
}

template<typename TensorBindingCollection>
void ArmnnPreparedModel::DumpTensorsIfRequired(char const* tensorNamePrefix,
                                               const TensorBindingCollection& tensorBindings) const
{
    if (!m_RequestInputsAndOutputsDumpDir.empty())
    {
        const std::string requestName = std::to_string(m_NetworkId) + ".dump";
        for (std::size_t i = 0u; i < tensorBindings.size(); ++i)
        {
            DumpTensor(m_RequestInputsAndOutputsDumpDir,
                       requestName,
                       BuildTensorName(tensorNamePrefix, i),
                       tensorBindings[i].second);
        }
    }
}

ArmnnPreparedModel::~ArmnnPreparedModel()
{
    VLOG(DRIVER) << "ArmnnPreparedModel::~ArmnnPreparedModel()";
    // Get a hold of the profiler used by this model.
    if (m_GpuProfilingEnabled)
    {
        auto profiler = m_Runtime->GetProfiler(m_NetworkId);
        if (profiler)
        {
            // Dump the profiling info to a file if required.
            DumpJsonProfilingIfRequired(m_GpuProfilingEnabled,
                                        m_RequestInputsAndOutputsDumpDir,
                                        m_NetworkId,
                                        profiler.get());
        }
    }
    // Unload the network associated with this model
    m_Runtime->UnloadNetwork(m_NetworkId);
}

bool ArmnnPreparedModel::ExecuteWithDummyInputs(unsigned int numInputs, unsigned int numOutputs) const
{
    std::vector<std::vector<char>> storage;
    armnn::InputTensors inputTensors;
    for (unsigned int i = 0; i < numInputs; i++)
    {
        armnn::TensorInfo inputTensorInfo = m_Runtime->GetInputTensorInfo(m_NetworkId, i);
        // pInputTensors (of type InputTensors) is composed of a vector of ConstTensors.
        // Therefore, set all TensorInfo isConstant parameters of input Tensors to true.
        inputTensorInfo.SetConstant();
        storage.emplace_back(inputTensorInfo.GetNumBytes());
        const armnn::ConstTensor inputTensor(inputTensorInfo, storage.back().data());

        inputTensors.emplace_back(i, inputTensor);
    }

    armnn::OutputTensors outputTensors;
    for (unsigned int i = 0; i < numOutputs; i++)
    {
        const armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkId, i);
        storage.emplace_back(outputTensorInfo.GetNumBytes());
        const armnn::Tensor outputTensor(outputTensorInfo, storage.back().data());

        outputTensors.emplace_back(i, outputTensor);
    }
    CanonicalExecutionContext ctx;
    ctx.measureTimings = MeasureTiming::NO;
    auto memPools = std::make_shared<std::vector<::android::nn::RunTimePoolInfo>>();

    auto errorStatus = ExecuteGraph(memPools,
                                    inputTensors,
                                    outputTensors,
                                    ctx);

    return errorStatus == ErrorStatus::NONE;
}

} // namespace armnn_driver
