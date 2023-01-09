//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <android-base/logging.h>
#include <nnapi/IBuffer.h>
#include <nnapi/IDevice.h>
#include <nnapi/IPreparedModel.h>
#include <nnapi/OperandTypes.h>
#include <nnapi/Result.h>
#include <nnapi/Types.h>
#include <nnapi/Validation.h>

#include "ArmnnDevice.hpp"
#include "ArmnnDriverImpl.hpp"
#include "Converter.hpp"

#include "ArmnnDriverImpl.hpp"
#include "ModelToINetworkTransformer.hpp"

#include <armnn/Version.hpp>
#include <log/log.h>
namespace armnn_driver
{

//using namespace android::nn;

class ArmnnDriver : public IDevice
{
private:
    std::unique_ptr<ArmnnDevice> m_Device;
public:
    ArmnnDriver(DriverOptions options)
    {
        try
        {
            VLOG(DRIVER) << "ArmnnDriver::ArmnnDriver()";
            m_Device = std::unique_ptr<ArmnnDevice>(new ArmnnDevice(std::move(options)));
        }
        catch (armnn::InvalidArgumentException& ex)
        {
            VLOG(DRIVER) << "ArmnnDevice failed to initialise: " << ex.what();
        }
        catch (...)
        {
            VLOG(DRIVER) << "ArmnnDevice failed to initialise with an unknown error";
        }
    }

public:

    const std::string& getName() const override
    {
        VLOG(DRIVER) << "ArmnnDriver::getName()";
        static const std::string name = "arm-armnn-sl";
        return name;
    }

    const std::string& getVersionString() const override
    {
        VLOG(DRIVER) << "ArmnnDriver::getVersionString()";
        static const std::string versionString = ARMNN_VERSION;
        return versionString;
    }

    Version getFeatureLevel() const override
    {
        VLOG(DRIVER) << "ArmnnDriver::getFeatureLevel()";
        return kVersionFeatureLevel6;
    }

    DeviceType getType() const override
    {
        VLOG(DRIVER) << "ArmnnDriver::getType()";
        return DeviceType::CPU;
    }

    const std::vector<Extension>& getSupportedExtensions() const override
    {
        VLOG(DRIVER) << "ArmnnDriver::getSupportedExtensions()";
        static const std::vector<Extension> extensions = {};
        return extensions;
    }

    const Capabilities& getCapabilities() const override
    {
        VLOG(DRIVER) << "ArmnnDriver::GetCapabilities()";
        return ArmnnDriverImpl::GetCapabilities(m_Device->m_Runtime);
    }

    std::pair<uint32_t, uint32_t> getNumberOfCacheFilesNeeded() const override
    {
        VLOG(DRIVER) << "ArmnnDriver::getNumberOfCacheFilesNeeded()";
        unsigned int numberOfCachedModelFiles = 0;
        for (auto& backend : m_Device->m_Options.GetBackends())
        {
            numberOfCachedModelFiles += GetNumberOfCacheFiles(backend);
            VLOG(DRIVER) << "ArmnnDriver::getNumberOfCacheFilesNeeded() = "
                         << std::to_string(numberOfCachedModelFiles);
        }
        return std::make_pair(numberOfCachedModelFiles, 1ul);
    }

    GeneralResult<void> wait() const override
    {
        VLOG(DRIVER) << "ArmnnDriver::wait()";
        return {};
    }

    GeneralResult<std::vector<bool>> getSupportedOperations(const Model& model) const override
    {
        VLOG(DRIVER) << "ArmnnDriver::getSupportedOperations()";
        if (m_Device.get() == nullptr)
        {
            return NN_ERROR(ErrorStatus::DEVICE_UNAVAILABLE) << "Device Unavailable!";
        }

        std::stringstream ss;
        ss << "ArmnnDriverImpl::getSupportedOperations()";
        std::string fileName;
        std::string timestamp;
        if (!m_Device->m_Options.GetRequestInputsAndOutputsDumpDir().empty())
        {
            ss << " : "
               << m_Device->m_Options.GetRequestInputsAndOutputsDumpDir()
               << "/"
               // << GetFileTimestamp()
               << "_getSupportedOperations.txt";
        }
        VLOG(DRIVER) << ss.str().c_str();

        if (!m_Device->m_Options.GetRequestInputsAndOutputsDumpDir().empty())
        {
            //dump the marker file
            std::ofstream fileStream;
            fileStream.open(fileName, std::ofstream::out | std::ofstream::trunc);
            if (fileStream.good())
            {
                fileStream << timestamp << std::endl;
                fileStream << timestamp << std::endl;
            }
            fileStream.close();
        }

        std::vector<bool> result;
        if (!m_Device->m_Runtime)
        {
            return NN_ERROR(ErrorStatus::DEVICE_UNAVAILABLE) << "Device Unavailable!";
        }

        // Run general model validation, if this doesn't pass we shouldn't analyse the model anyway.
        if (const auto result = validate(model); !result.ok())
        {
            return NN_ERROR(ErrorStatus::INVALID_ARGUMENT) << "Invalid Model!";
        }

        // Attempt to convert the model to an ArmNN input network (INetwork).
        ModelToINetworkTransformer modelConverter(m_Device->m_Options.GetBackends(),
                                                  model,
                                                  m_Device->m_Options.GetForcedUnsupportedOperations());

        if (modelConverter.GetConversionResult() != ConversionResult::Success
            && modelConverter.GetConversionResult() != ConversionResult::UnsupportedFeature)
        {
            return NN_ERROR(ErrorStatus::GENERAL_FAILURE) << "Conversion Error!";
        }

        // Check each operation if it was converted successfully and copy the flags
        // into the result (vector<bool>) that we need to return to Android.
        result.reserve(model.main.operations.size());
        for (uint32_t operationIdx = 0; operationIdx < model.main.operations.size(); ++operationIdx)
        {
            bool operationSupported = modelConverter.IsOperationSupported(operationIdx);
            result.push_back(operationSupported);
        }

        return result;
    }

    GeneralResult<SharedPreparedModel> prepareModel(const Model& model,
        ExecutionPreference preference,
        Priority priority,
        OptionalTimePoint deadline,
        const std::vector<SharedHandle>& modelCache,
        const std::vector<SharedHandle>& dataCache,
        const CacheToken& token,
        const std::vector<android::nn::TokenValuePair>& hints,
        const std::vector<android::nn::ExtensionNameAndPrefix>& extensionNameToPrefix) const override
    {
        VLOG(DRIVER) << "ArmnnDriver::prepareModel()";

        if (m_Device.get() == nullptr)
        {
            return NN_ERROR(ErrorStatus::DEVICE_UNAVAILABLE) << "Device Unavailable!";
        }
        // Validate arguments.
        if (const auto result = validate(model); !result.ok()) {
            return NN_ERROR(ErrorStatus::INVALID_ARGUMENT) << "Invalid Model: " << result.error();
        }
        if (const auto result = validate(preference); !result.ok()) {
            return NN_ERROR(ErrorStatus::INVALID_ARGUMENT)
                << "Invalid ExecutionPreference: " << result.error();
        }
        if (const auto result = validate(priority); !result.ok()) {
            return NN_ERROR(ErrorStatus::INVALID_ARGUMENT) << "Invalid Priority: " << result.error();
        }

        // Check if deadline has passed.
        if (hasDeadlinePassed(deadline)) {
            return NN_ERROR(ErrorStatus::MISSED_DEADLINE_PERSISTENT);
        }

        return ArmnnDriverImpl::PrepareArmnnModel(m_Device->m_Runtime,
            m_Device->m_ClTunedParameters,
            m_Device->m_Options,
            model,
            modelCache,
            dataCache,
            token,
            model.relaxComputationFloat32toFloat16 && m_Device->m_Options.GetFp16Enabled(),
            priority);
    }

    GeneralResult<SharedPreparedModel> prepareModelFromCache(OptionalTimePoint deadline,
                                                             const std::vector<SharedHandle>& modelCache,
                                                             const std::vector<SharedHandle>& dataCache,
                                                             const CacheToken& token) const override
    {
        VLOG(DRIVER) << "ArmnnDriver::prepareModelFromCache()";
        if (m_Device.get() == nullptr)
        {
            return NN_ERROR(ErrorStatus::DEVICE_UNAVAILABLE) << "Device Unavailable!";
        }
        // Check if deadline has passed.
        if (hasDeadlinePassed(deadline)) {
            return NN_ERROR(ErrorStatus::MISSED_DEADLINE_PERSISTENT);
        }

        return ArmnnDriverImpl::PrepareArmnnModelFromCache(
                     m_Device->m_Runtime,
                     m_Device->m_ClTunedParameters,
                     m_Device->m_Options,
                     modelCache,
                     dataCache,
                     token,
                     m_Device->m_Options.GetFp16Enabled());
    }

    GeneralResult<SharedBuffer> allocate(const BufferDesc&,
                                         const std::vector<SharedPreparedModel>&,
                                         const std::vector<BufferRole>&,
                                         const std::vector<BufferRole>&) const override
    {
        VLOG(DRIVER) << "ArmnnDriver::allocate()";
        return NN_ERROR(ErrorStatus::INVALID_ARGUMENT) << "ArmnnDriver::allocate -- does not support allocate.";
    }
};

} // namespace armnn_driver
