//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArmnnDriverImpl.hpp"
#include "ArmnnPreparedModel.hpp"
#include "ModelToINetworkTransformer.hpp"
#include "SystemPropertiesUtils.hpp"

#include <armnnDeserializer/IDeserializer.hpp>

#include <log/log.h>
#include <sys/stat.h>

namespace
{

Capabilities GenerateCapabilities()
{
    VLOG(DRIVER) << "ArmnnDriverImpl::GenerateCapabilities()";

    float defaultPerfValue = .1f;
    const Capabilities::PerformanceInfo defaultPerfInfo = { /* execTime */ defaultPerfValue,
                                                            /* powerUsage */ defaultPerfValue
                                                          };
    std::vector<OperandType> operandsTypes({
                OperandType::FLOAT32,
                OperandType::INT32,
                OperandType::UINT32,
                OperandType::TENSOR_FLOAT32,
                OperandType::TENSOR_INT32,
                OperandType::TENSOR_QUANT8_ASYMM,
                OperandType::BOOL,
                OperandType::TENSOR_QUANT16_SYMM,
                OperandType::TENSOR_FLOAT16,
                OperandType::TENSOR_BOOL8,
                OperandType::FLOAT16,
                OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL,
                OperandType::TENSOR_QUANT16_ASYMM,
                OperandType::TENSOR_QUANT8_SYMM,
                OperandType::TENSOR_QUANT8_ASYMM_SIGNED,
    });

    std::vector<Capabilities::OperandPerformance> operandPerformances;
    operandPerformances.reserve(operandsTypes.size());

    for (auto opType : operandsTypes)
    {
        operandPerformances.push_back(
                Capabilities::OperandPerformance{ /* type */ opType, /* info */ defaultPerfInfo });
    }

    auto operandPerformanceTable =
               Capabilities::OperandPerformanceTable::create(std::move(operandPerformances)).value();

    return { /* relaxedFloat32toFloat16PerformanceScalar */ defaultPerfInfo,
             /* relaxedFloat32toFloat16PerformanceTensor */ defaultPerfInfo,
             /* operandPerformance */ std::move(operandPerformanceTable),
             /* ifPerformance */ defaultPerfInfo,
             /* whilePerformance */ defaultPerfInfo };
}

size_t Hash(std::vector<uint8_t>& cacheData)
{
    std::size_t hash = cacheData.size();
    for (auto& i : cacheData)
    {
        hash = ((hash << 5) - hash) + i;
    }
    return hash;
}

} // anonymous namespace

using namespace android::nn;

namespace armnn_driver
{

bool ArmnnDriverImpl::ValidateSharedHandle(const SharedHandle& sharedHandle)
{
    bool valid = true;

    if (*sharedHandle < 0)
    {
        return !valid;
    }

    int dataCacheFileAccessMode = fcntl(*sharedHandle, F_GETFL) & O_ACCMODE;
    if (dataCacheFileAccessMode != O_RDWR)
    {
        return !valid;
    }

    return valid;
}

GeneralResult<SharedPreparedModel> ArmnnDriverImpl::PrepareArmnnModel(
    const armnn::IRuntimePtr& runtime,
    const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
    const DriverOptions& options,
    const Model& model,
    const std::vector<SharedHandle>& modelCacheHandle,
    const std::vector<SharedHandle>& dataCacheHandle,
    const CacheToken& token,
    bool float32ToFloat16,
    Priority priority)
{
    VLOG(DRIVER) << "ArmnnDriverImpl::PrepareArmnnModel()";

    if (!runtime)
    {
        return NN_ERROR(ErrorStatus::DEVICE_UNAVAILABLE) << "Device unavailable";
    }

    if (const auto result = validate(model); !result.ok())
    {
        return NN_ERROR(ErrorStatus::INVALID_ARGUMENT) << "Invalid model passed as input";
    }

    // Deliberately ignore any unsupported operations requested by the options -
    // at this point we're being asked to prepare a model that we've already declared support for
    // and the operation indices may be different to those in getSupportedOperations anyway.
    std::set<unsigned int> unsupportedOperations;
    ModelToINetworkTransformer modelConverter(options.GetBackends(),
                                              model,
                                              unsupportedOperations);

    if (modelConverter.GetConversionResult() != ConversionResult::Success)
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << "ModelToINetworkConverter failed";
    }

    // Serialize the network graph to a .armnn file if an output directory
    // has been specified in the drivers' arguments.
    std::vector<uint8_t> dataCacheData;
    bool serializeToFile = dataCacheHandle.size() < 1 ? false : true;
    auto serializedNetworkFileName =
            SerializeNetwork(*modelConverter.GetINetwork(),
                             options.GetRequestInputsAndOutputsDumpDir(),
                             dataCacheData,
                             serializeToFile);

    // Optimize the network
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    armnn::OptimizerOptionsOpaque OptOptions;
    OptOptions.SetReduceFp32ToFp16(float32ToFloat16);
    OptOptions.SetProfilingEnabled(options.IsGpuProfilingEnabled());

    int cachedFd = -1;
    bool saveCachedNetwork = options.SaveCachedNetwork();

    unsigned int numberOfCachedModelFiles = 0;
    if (modelCacheHandle.size() > 0)
    {
        unsigned int index = 0;
        for (auto& backend : options.GetBackends())
        {
            // modelCacheHandle size should be equal to numberOfCachedModelFiles
            // modelCacheHandle vector should be in same order as backends
            auto numberOfCacheFiles = GetNumberOfCacheFiles(backend);
            if (numberOfCacheFiles > 0)
            {
                numberOfCachedModelFiles += numberOfCacheFiles;
                // For GpuAcc numberOfCachedFiles is 1
                if (backend == armnn::Compute::GpuAcc)
                {
                    cachedFd = *modelCacheHandle[index];
                    saveCachedNetwork = true;
                }
                index += numberOfCachedModelFiles;
            }
        }
    }

    armnn::BackendOptions gpuAcc("GpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() },
        { "SaveCachedNetwork", saveCachedNetwork },
        { "CachedNetworkFilePath", options.GetCachedNetworkFilePath() },
        { "MLGOTuningFilePath", options.GetClMLGOTunedParametersFile() },
        { "CachedFileDescriptor", cachedFd }
    });

    armnn::BackendOptions cpuAcc("CpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() },
        { "NumberOfThreads", options.GetNumberOfThreads() }
    });
    OptOptions.AddModelOption(gpuAcc);
    OptOptions.AddModelOption(cpuAcc);

    std::vector<std::string> errMessages;
    try
    {
        optNet = armnn::Optimize(*modelConverter.GetINetwork(),
                                 options.GetBackends(),
                                 runtime->GetDeviceSpec(),
                                 OptOptions,
                                 errMessages);
    }
    catch (std::exception& e)
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << e.what();
    }

    // Check that the optimized network is valid.
    if (!optNet)
    {
        std::stringstream message;
        message << "Invalid optimized network";
        for (const std::string& msg : errMessages)
        {
            message << "\n" << msg;
        }
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << message.str();
    }

    // Export the optimized network graph to a dot file if an output dump directory
    // has been specified in the drivers' arguments.
    std::string dotGraphFileName = ExportNetworkGraphToDotFile(*optNet,
                                                               options.GetRequestInputsAndOutputsDumpDir());

    // Load it into the runtime.
    armnn::NetworkId netId = 0;
    std::string msg;
    armnn::INetworkProperties networkProperties(options.isAsyncModelExecutionEnabled(),
                                                MemorySource::Undefined,
                                                MemorySource::Undefined,
                                                options.IsGpuProfilingEnabled());
    auto numInputs  = getMainModel(model).inputIndexes.size();
    auto numOutputs = getMainModel(model).outputIndexes.size();
    try
    {
        if (runtime->LoadNetwork(netId, move(optNet), msg, networkProperties) != armnn::Status::Success)
        {
            return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << "Network could not be loaded";
        }
    }
    catch (std::exception& e)
    {
        std::stringstream message;
        message << "Exception (" << e.what()<< ") caught from LoadNetwork.";
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << message.str();
    }

    // Now that we have a networkId for the graph rename the exported files to use it
    // so that we can associate the graph file and the input/output tensor exported files
    RenameExportedFiles(serializedNetworkFileName,
                        dotGraphFileName,
                        options.GetRequestInputsAndOutputsDumpDir(),
                        netId);

    // Cache the model
    size_t hashValue = 0;
    if (dataCacheHandle.size() == 1 )
    {
        hashValue = Hash(dataCacheData);
    }

    // Cache the model data
    if (modelCacheHandle.size() > 0)
    {
        if (modelCacheHandle.size() == numberOfCachedModelFiles)
        {
            for (uint32_t i = 0; i < modelCacheHandle.size(); ++i)
            {
                int modelCacheFileAccessMode = fcntl(*modelCacheHandle[i], F_GETFL) & O_ACCMODE;
                if (modelCacheFileAccessMode != O_RDONLY)
                {
                    struct stat statBuffer;
                    if (fstat(*modelCacheHandle[i], &statBuffer) == 0)
                    {
                        long modelDataSize = statBuffer.st_size;
                        if (modelDataSize > 0)
                        {
                            std::vector<uint8_t> modelData(modelDataSize);
                            pread(*modelCacheHandle[i], modelData.data(), modelData.size(), 0);
                            hashValue ^= Hash(modelData);
                        }
                    }
                }
            }
        }
    }
    if (dataCacheHandle.size() == 1 && hashValue != 0)
    {
        std::vector<uint8_t> theHashValue(sizeof(hashValue));
        ::memcpy(theHashValue.data(), &hashValue, sizeof(hashValue));

        write(*dataCacheHandle[0], theHashValue.data(), theHashValue.size());
        pwrite(*dataCacheHandle[0], dataCacheData.data(), dataCacheData.size(), theHashValue.size());
    }

    bool executeWithDummyInputs = (std::find(options.GetBackends().begin(),
                                            options.GetBackends().end(),
                                            armnn::Compute::GpuAcc) != options.GetBackends().end());

    auto preparedModel = std::make_shared<const ArmnnPreparedModel>(netId,
                                                                    runtime.get(),
                                                                    model,
                                                                    options.GetRequestInputsAndOutputsDumpDir(),
                                                                    options.IsGpuProfilingEnabled(),
                                                                    priority);

    // Run a single 'dummy' inference of the model. This means that CL kernels will get compiled (and tuned if
    // this is enabled) before the first 'real' inference which removes the overhead of the first inference.
    // Only run this if the GpuAcc backend has been added to options
    if (std::find(options.GetBackends().begin(),
                  options.GetBackends().end(),
                  armnn::Compute::GpuAcc) != options.GetBackends().end())
    {
        if (!preparedModel->ExecuteWithDummyInputs(numInputs, numOutputs))
        {
            return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << "Network could not be executed";
        }

        if (clTunedParameters &&
            options.GetClTunedParametersMode() == armnn::IGpuAccTunedParameters::Mode::UpdateTunedParameters)
        {
            // Now that we've done one inference the CL kernel parameters will have been tuned,
            // so save the updated file.
            try
            {
                clTunedParameters->Save(options.GetClTunedParametersFile().c_str());
            }
            catch (std::exception& error)
            {
                VLOG(DRIVER) << "ArmnnDriverImpl::prepareModel: Failed to save CL tuned parameters file"
                             << options.GetClTunedParametersFile().c_str() << error.what();
            }
        }
    }
    return std::move(preparedModel);
}

GeneralResult<SharedPreparedModel> ArmnnDriverImpl::PrepareArmnnModelFromCache(
    const armnn::IRuntimePtr& runtime,
    const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
    const DriverOptions& options,
    const std::vector<SharedHandle>& modelCacheHandle,
    const std::vector<SharedHandle>& dataCacheHandle,
    const CacheToken& token,
    bool float32ToFloat16)
{
    VLOG(DRIVER) << "ArmnnDriverImpl::PrepareArmnnModelFromCache()";

    if (!runtime)
    {
        return NN_ERROR(ErrorStatus::DEVICE_UNAVAILABLE)
                            << "ArmnnDriverImpl::prepareModelFromCache(): Device unavailable";
    }

    if (token.size() != ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN)
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE)
                            << "ArmnnDriverImpl::prepareModelFromCache(): Token size does not match!";
    }

    // Validate dataCacheHandle
    if (dataCacheHandle.size() != 1)
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE)
                            << "ArmnnDriverImpl::prepareModelFromCache(): Not valid data cache handle!";
    }

    if (!ValidateSharedHandle(dataCacheHandle[0]))
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE)
                << "ArmnnDriverImpl::prepareModelFromCache(): Not valid data cache handle!";
    }

    size_t cachedDataSize = 0;
    struct stat dataStatBuffer;
    if (fstat(*dataCacheHandle[0], &dataStatBuffer) == 0)
    {
        cachedDataSize = dataStatBuffer.st_size;
    }
    if (cachedDataSize == 0)
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE)
                << "ArmnnDriverImpl::prepareModelFromCache(): Not valid cached data!";
    }

    // Check if model files cached they match the expected value
    unsigned int numberOfCachedModelFiles = 0;
    for (auto& backend : options.GetBackends())
    {
        numberOfCachedModelFiles += GetNumberOfCacheFiles(backend);
    }
    if (modelCacheHandle.size() != numberOfCachedModelFiles)
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE)
                           << "ArmnnDriverImpl::prepareModelFromCache(): Model cache handle size does not match.";
    }

    // Read the hashValue
    std::vector<uint8_t> hashValue(sizeof(size_t));
    pread(*dataCacheHandle[0], hashValue.data(), hashValue.size(), 0);

    // Read the model
    if (cachedDataSize < hashValue.size())
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE)
                << "ArmnnDriverImpl::prepareModelFromCache(): cachedDataSize is less than hashValue!";
    }
    std::vector<uint8_t> dataCacheData(cachedDataSize - hashValue.size());
    pread(*dataCacheHandle[0], dataCacheData.data(), dataCacheData.size(), hashValue.size());
    auto calculatedHashValue = Hash(dataCacheData);

    int gpuAccCachedFd = -1;
    if (modelCacheHandle.size() > 0)
    {
        unsigned int index = 0;
        for (auto& backend : options.GetBackends())
        {
            // modelCacheHandle size should be equal to numberOfCachedModelFiles
            // modelCacheHandle vector should be in same order as backends
            auto numberOfCacheFiles = GetNumberOfCacheFiles(backend);
            if (numberOfCacheFiles > 0)
            {
                if (!ValidateSharedHandle(modelCacheHandle[index]))
                {
                    return NN_ERROR(ErrorStatus::GENERAL_FAILURE)
                            << "ArmnnDriverImpl::prepareModelFromCache(): Invalid model cache handle!";
                }
                int cachedFd = *modelCacheHandle[index];
                struct stat statBuffer;
                if (fstat(cachedFd, &statBuffer) == 0)
                {
                    long modelDataSize = statBuffer.st_size;
                    if (modelDataSize > 0)
                    {
                        std::vector<uint8_t> modelData(modelDataSize);
                        pread(cachedFd, modelData.data(), modelData.size(), 0);
                        calculatedHashValue ^= Hash(modelData);

                        if (backend == armnn::Compute::GpuAcc)
                        {
                            gpuAccCachedFd = cachedFd;
                        }
                    }
                }
                index += numberOfCacheFiles;
            }
        }
    }

    std::vector<uint8_t> calculatedHashData(sizeof(calculatedHashValue));
    ::memcpy(calculatedHashData.data(), &calculatedHashValue, sizeof(calculatedHashValue));
    if (hashValue != calculatedHashData)
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE)
                << "ArmnnDriverImpl::prepareModelFromCache(): ValidateHash() failed!";
    }

    // Deserialize the network..
    armnn::INetworkPtr network = armnn::INetworkPtr(nullptr, [](armnn::INetwork*){});
    try
    {
        network = armnnDeserializer::IDeserializer::Create()->CreateNetworkFromBinary(dataCacheData);
    }
    catch (std::exception&)
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE)
                << "ArmnnDriverImpl::prepareModelFromCache(): Exception caught from Deserializer!";
    }

    // Optimize the network
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    armnn::OptimizerOptionsOpaque OptOptions;
    OptOptions.SetReduceFp32ToFp16(float32ToFloat16);
    OptOptions.SetProfilingEnabled(options.IsGpuProfilingEnabled());

    armnn::BackendOptions gpuAcc("GpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() },
        { "SaveCachedNetwork", false },
        { "CachedNetworkFilePath", options.GetCachedNetworkFilePath() },
        { "MLGOTuningFilePath", options.GetClMLGOTunedParametersFile() },
        { "CachedFileDescriptor", gpuAccCachedFd }
    });

    armnn::BackendOptions cpuAcc("CpuAcc",
    {
        { "FastMathEnabled", options.IsFastMathEnabled() },
        { "NumberOfThreads", options.GetNumberOfThreads() }
    });
    OptOptions.AddModelOption(gpuAcc);
    OptOptions.AddModelOption(cpuAcc);

    std::vector<std::string> errMessages;
    try
    {
        optNet = armnn::Optimize(*network.get(),
                                 options.GetBackends(),
                                 runtime->GetDeviceSpec(),
                                 OptOptions,
                                 errMessages);
    }
    catch (std::exception& e)
    {
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << e.what();
    }

    // Check that the optimized network is valid.
    if (!optNet)
    {
        std::stringstream message;
        message << "Invalid optimized network";
        for (const std::string& msg : errMessages)
        {
            message << "\n" << msg;
        }
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << message.str();
    }

    // Export the optimized network graph to a dot file if an output dump directory
    // has been specified in the drivers' arguments.
    std::string dotGraphFileName = ExportNetworkGraphToDotFile(*optNet,
                                                               options.GetRequestInputsAndOutputsDumpDir());

    // Load it into the runtime.
    armnn::NetworkId netId = 0;
    std::string msg;
    armnn::INetworkProperties networkProperties(options.isAsyncModelExecutionEnabled(),
                                                MemorySource::Undefined,
                                                MemorySource::Undefined,
                                                options.IsGpuProfilingEnabled());
    try
    {
        if (runtime->LoadNetwork(netId, move(optNet), msg, networkProperties) != armnn::Status::Success)
        {
            return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << "Network could not be loaded";
        }
    }
    catch (std::exception& e)
    {
        std::stringstream message;
        message << "Exception (" << e.what()<< ") caught from LoadNetwork.";
        return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << message.str();
    }

    auto preparedModel = std::make_shared<const ArmnnPreparedModel>(netId,
                                                      runtime.get(),
                                                      options.GetRequestInputsAndOutputsDumpDir(),
                                                      options.IsGpuProfilingEnabled(),
                                                      Priority::MEDIUM,
                                                      true);
    return std::move(preparedModel);
}

const Capabilities& ArmnnDriverImpl::GetCapabilities(const armnn::IRuntimePtr& runtime)
{
    VLOG(DRIVER) << "ArmnnDriverImpl::GetCapabilities()";
    static const Capabilities theCapabilities = GenerateCapabilities();
    return theCapabilities;
}

} // namespace armnn_driver
