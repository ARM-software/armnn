//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ExecuteNetworkParams.hpp"

#include "NetworkExecutionUtils/NetworkExecutionUtils.hpp"
#include <armnn/Logging.hpp>

#include <fmt/format.h>
#include <armnnUtils/Filesystem.hpp>

void CheckClTuningParameter(const int& tuningLevel,
                            const std::string& tuningPath,
                            const std::vector<armnn::BackendId> computeDevices)
{
    if (!tuningPath.empty())
    {
        if (tuningLevel == 0)
        {
            ARMNN_LOG(info) << "Using cl tuning file: " << tuningPath << "\n";
            if (!ValidatePath(tuningPath, true))
            {
                throw armnn::InvalidArgumentException("The tuning path is not valid");
            }
        }
        else if ((1 <= tuningLevel) && (tuningLevel <= 3))
        {
            ARMNN_LOG(info) << "Starting execution to generate a cl tuning file: " << tuningPath << "\n"
                            << "Tuning level in use: " << tuningLevel << "\n";
        }
        else if ((0 < tuningLevel) || (tuningLevel > 3))
        {
            throw armnn::InvalidArgumentException(fmt::format("The tuning level {} is not valid.",
                                                              tuningLevel));
        }

        // Ensure that a GpuAcc is enabled. Otherwise no tuning data are used or genereted
        // Only warn if it's not enabled
        auto it = std::find(computeDevices.begin(), computeDevices.end(), "GpuAcc");
        if (it == computeDevices.end())
        {
            ARMNN_LOG(warning) << "To use Cl Tuning the compute device GpuAcc needs to be active.";
        }
    }
}

void ExecuteNetworkParams::ValidateParams()
{
    if (m_DynamicBackendsPath == "")
    {
        // Check compute devices are valid unless they are dynamically loaded at runtime
        std::string invalidBackends;
        if (!CheckRequestedBackendsAreValid(m_ComputeDevices, armnn::Optional<std::string&>(invalidBackends)))
        {
            ARMNN_LOG(fatal) << "The list of preferred devices contains invalid backend IDs: "
                             << invalidBackends;
        }
    }
    CheckClTuningParameter(m_TuningLevel, m_TuningPath, m_ComputeDevices);

    if (m_EnableBf16TurboMode && m_EnableFp16TurboMode)
    {
        throw armnn::InvalidArgumentException("BFloat16 and Float16 turbo mode cannot be "
                                              "enabled at the same time.");
    }

    // Check input tensor shapes
    if ((m_InputTensorShapes.size() != 0) &&
        (m_InputTensorShapes.size() != m_InputNames.size()))
    {
        throw armnn::InvalidArgumentException("input-name and input-tensor-shape must have "
                                              "the same amount of elements. ");
    }

    if (m_InputTensorDataFilePaths.size() != 0)
    {
        if (!ValidatePaths(m_InputTensorDataFilePaths, true))
        {
            throw armnn::InvalidArgumentException("One or more input data file paths are not valid.");
        }

        if (m_InputTensorDataFilePaths.size() < m_InputNames.size())
        {
            throw armnn::InvalidArgumentException(
                    fmt::format("According to the number of input names the user provided the network has {} "
                                "inputs. But only {} input-tensor-data file paths were provided. Each input of the "
                                "model is expected to be stored in it's own file.",
                                m_InputNames.size(),
                                m_InputTensorDataFilePaths.size()));
        }
    }

    // Check that threshold time is not less than zero
    if (m_ThresholdTime < 0)
    {
        throw armnn::InvalidArgumentException("Threshold time supplied as a command line argument is less than zero.");
    }

    // Warn if ExecuteNetwork will generate dummy input data
    if (m_GenerateTensorData)
    {
        ARMNN_LOG(warning) << "No input files provided, input tensors will be filled with 0s.";
    }

    if (m_AllowExpandedDims && m_InferOutputShape)
    {
        throw armnn::InvalidArgumentException("infer-output-shape and allow-expanded-dims cannot be used together.");
    }
}

#if defined(ARMNN_TFLITE_DELEGATE)
/**
 * A utility method that populates a DelegateOptions object from this ExecuteNetworkParams.
 *
 * @return a populated armnnDelegate::DelegateOptions object.
 */
armnnDelegate::DelegateOptions ExecuteNetworkParams::ToDelegateOptions() const
{
    armnnDelegate::DelegateOptions delegateOptions(m_ComputeDevices);
    delegateOptions.SetDynamicBackendsPath(m_DynamicBackendsPath);
    delegateOptions.SetGpuProfilingState(m_EnableProfiling);

    armnn::OptimizerOptions options;
    options.m_ReduceFp32ToFp16 = m_EnableFp16TurboMode;
    options.m_ReduceFp32ToBf16 = m_EnableBf16TurboMode;
    options.m_Debug = m_PrintIntermediate;
    options.m_DebugToFile = m_PrintIntermediateOutputsToFile;
    options.m_ProfilingEnabled = m_EnableProfiling;
    delegateOptions.SetInternalProfilingParams(m_EnableProfiling, armnn::ProfilingDetailsMethod::DetailsWithEvents);
    options.m_shapeInferenceMethod = armnn::ShapeInferenceMethod::ValidateOnly;
    if (m_InferOutputShape)
    {
        options.m_shapeInferenceMethod = armnn::ShapeInferenceMethod::InferAndValidate;
    }

    armnn::BackendOptions gpuAcc("GpuAcc",
                                 {
        { "FastMathEnabled", m_EnableFastMath },
        { "SaveCachedNetwork", m_SaveCachedNetwork },
        { "CachedNetworkFilePath", m_CachedNetworkFilePath },
        { "TuningLevel", m_TuningLevel},
        { "TuningFile", m_TuningPath.c_str()},
        { "KernelProfilingEnabled", m_EnableProfiling},
        { "MLGOTuningFilePath", m_MLGOTuningFilePath}
                                 });

    armnn::BackendOptions cpuAcc("CpuAcc",
                                 {
        { "FastMathEnabled", m_EnableFastMath },
        { "NumberOfThreads", m_NumberOfThreads }
                                 });
    options.m_ModelOptions.push_back(gpuAcc);
    options.m_ModelOptions.push_back(cpuAcc);

    if (m_InferOutputShape)
    {
        armnn::BackendOptions networkOption("ShapeInferenceMethod",
                                            {
                                                    {"InferAndValidate", true}
                                            });
        options.m_ModelOptions.push_back(networkOption);
    }
    if (m_AllowExpandedDims)
    {
        armnn::BackendOptions networkOption("AllowExpandedDims",
                                            {
                                                    {"AllowExpandedDims", true}
                                            });
        options.m_ModelOptions.push_back(networkOption);
    }
    delegateOptions.SetOptimizerOptions(options);

    return delegateOptions;
}

#endif
