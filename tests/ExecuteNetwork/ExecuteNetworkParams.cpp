//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
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

    if (m_EnableBf16TurboMode && !m_EnableFastMath)
    {
        throw armnn::InvalidArgumentException("To use BF16 please use --enable-fast-math. ");
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
    delegateOptions.SetInternalProfilingParams(m_EnableProfiling, armnn::ProfilingDetailsMethod::DetailsWithEvents);

    // GPU Backend options first.
    {
        armnn::BackendOptions gpuOption("GpuAcc", {{"TuningLevel", m_TuningLevel}});
        delegateOptions.AddBackendOption(gpuOption);
    }
    {
        armnn::BackendOptions gpuOption("GpuAcc", {{"TuningFile", m_TuningPath.c_str()}});
        delegateOptions.AddBackendOption(gpuOption);
    }
    {
        armnn::BackendOptions gpuOption("GpuAcc", {{"KernelProfilingEnabled", m_EnableProfiling}});
        delegateOptions.AddBackendOption(gpuOption);
    }

    // Optimizer options next.
    armnn::OptimizerOptionsOpaque optimizerOptions;
    optimizerOptions.SetReduceFp32ToFp16(m_EnableFp16TurboMode);
    optimizerOptions.SetDebugEnabled(m_PrintIntermediate);
    optimizerOptions.SetDebugToFileEnabled(m_PrintIntermediateOutputsToFile);
    optimizerOptions.SetProfilingEnabled(m_EnableProfiling);
    optimizerOptions.SetShapeInferenceMethod(armnn::ShapeInferenceMethod::ValidateOnly);
    if (m_InferOutputShape)
    {
        optimizerOptions.SetShapeInferenceMethod(armnn::ShapeInferenceMethod::InferAndValidate);
        armnn::BackendOptions networkOption("ShapeInferenceMethod",
                                            {
                                                {"InferAndValidate", true}
                                            });
        optimizerOptions.AddModelOption(networkOption);
    }

    {
        armnn::BackendOptions option("GpuAcc", {{"FastMathEnabled", m_EnableFastMath}});
        optimizerOptions.AddModelOption(option);
    }
    {
        armnn::BackendOptions option("GpuAcc", {{"CachedNetworkFilePath", m_CachedNetworkFilePath}});
        optimizerOptions.AddModelOption(option);
    }
    {
        armnn::BackendOptions option("GpuAcc", {{"MLGOTuningFilePath", m_MLGOTuningFilePath}});
        optimizerOptions.AddModelOption(option);
    }

    armnn::BackendOptions cpuAcc("CpuAcc",
                                 {
        { "FastMathEnabled", m_EnableFastMath },
        { "NumberOfThreads", m_NumberOfThreads }
                                 });
    optimizerOptions.AddModelOption(cpuAcc);
    if (m_AllowExpandedDims)
    {
        armnn::BackendOptions networkOption("AllowExpandedDims",
                                            {
                                                    {"AllowExpandedDims", true}
                                            });
        optimizerOptions.AddModelOption(networkOption);
    }
    delegateOptions.SetOptimizerOptions(optimizerOptions);
    return delegateOptions;
}

#endif
