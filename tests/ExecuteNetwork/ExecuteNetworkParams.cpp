//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ExecuteNetworkParams.hpp"

#include "NetworkExecutionUtils/NetworkExecutionUtils.hpp"
#include <InferenceModel.hpp>
#include <armnn/Logging.hpp>

#include <fmt/format.h>

bool IsModelBinary(const std::string& modelFormat)
{
    // Parse model binary flag from the model-format string we got from the command-line
    if (modelFormat.find("binary") != std::string::npos)
    {
        return true;
    }
    else if (modelFormat.find("txt") != std::string::npos || modelFormat.find("text") != std::string::npos)
    {
        return false;
    }
    else
    {
        throw armnn::InvalidArgumentException(fmt::format("Unknown model format: '{}'. "
                                                          "Please include 'binary' or 'text'",
                                                          modelFormat));
    }
}

void CheckModelFormat(const std::string& modelFormat)
{
    // Forward to implementation based on the parser type
    if (modelFormat.find("armnn") != std::string::npos)
    {
#if defined(ARMNN_SERIALIZER)
#else
        throw armnn::InvalidArgumentException("Can't run model in armnn format without a "
                                              "built with serialization support.");
#endif
    }
    else if (modelFormat.find("onnx") != std::string::npos)
    {
#if defined(ARMNN_ONNX_PARSER)
#else
        throw armnn::InvalidArgumentException("Can't run model in onnx format without a "
                                              "built with Onnx parser support.");
#endif
    }
    else if (modelFormat.find("tflite") != std::string::npos)
    {
#if defined(ARMNN_TF_LITE_PARSER)
        if (!IsModelBinary(modelFormat))
        {
            throw armnn::InvalidArgumentException(fmt::format("Unknown model format: '{}'. Only 'binary' "
                                                              "format supported for tflite files",
                                                              modelFormat));
        }
#elif defined(ARMNN_TFLITE_DELEGATE)
#else
        throw armnn::InvalidArgumentException("Can't run model in tflite format without a "
                                              "built with Tensorflow Lite parser support.");
#endif
    }
    else
    {
        throw armnn::InvalidArgumentException(fmt::format("Unknown model format: '{}'. "
                                                          "Please include 'tflite' or 'onnx'",
                                                          modelFormat));
    }
}

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

    m_IsModelBinary = IsModelBinary(m_ModelFormat);

    CheckModelFormat(m_ModelFormat);

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
        else if (m_InputTensorDataFilePaths.size() % m_InputNames.size() != 0)
        {
            throw armnn::InvalidArgumentException(
                    fmt::format("According to the number of input names the user provided the network has {} "
                                "inputs. The user specified {} input-tensor-data file paths which is not "
                                "divisible by the number of inputs.",
                                m_InputNames.size(),
                                m_InputTensorDataFilePaths.size()));
        }
    }

    if (m_InputTypes.size() == 0)
    {
        //Defaults the value of all inputs to "float"
        m_InputTypes.assign(m_InputNames.size(), "float");
    }
    else if ((m_InputTypes.size() != 0) &&
             (m_InputTypes.size() != m_InputNames.size()))
    {
        throw armnn::InvalidArgumentException("input-name and input-type must have the same amount of elements.");
    }

    // Make sure that the number of input files given is divisible by the number of inputs of the model
    if (!(m_InputTensorDataFilePaths.size() % m_InputNames.size() == 0))
    {
        throw armnn::InvalidArgumentException(
                fmt::format("The number of input-tensor-data files ({0}) is not divisible by the "
                            "number of inputs ({1} according to the number of input names).",
                            m_InputTensorDataFilePaths.size(),
                            m_InputNames.size()));
    }

    if (m_OutputTypes.size() == 0)
    {
        //Defaults the value of all outputs to "float"
        m_OutputTypes.assign(m_OutputNames.size(), "float");
    }
    else if ((m_OutputTypes.size() != 0) &&
             (m_OutputTypes.size() != m_OutputNames.size()))
    {
        throw armnn::InvalidArgumentException("output-name and output-type must have the same amount of elements.");
    }

    // Make sure that the number of output files given is equal to the number of outputs of the model
    // or equal to the number of outputs of the model multiplied with the number of iterations
    if (!m_OutputTensorFiles.empty())
    {
        if ((m_OutputTensorFiles.size() != m_OutputNames.size()) &&
            (m_OutputTensorFiles.size() != m_OutputNames.size() * m_Iterations))
        {
            std::stringstream errmsg;
            auto numOutputs = m_OutputNames.size();
            throw armnn::InvalidArgumentException(
                    fmt::format("The user provided {0} output-tensor files. The only allowed number of output-tensor "
                                "files is the number of outputs of the network ({1} according to the number of "
                                "output names) or the number of outputs multiplied with the number of times the "
                                "network should be executed (NumOutputs * NumIterations = {1} * {2} = {3}).",
                                m_OutputTensorFiles.size(),
                                numOutputs,
                                m_Iterations,
                                numOutputs*m_Iterations));
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

    delegateOptions.SetOptimizerOptions(options);

    // If v,visualize-optimized-model is enabled then construct a file name for the dot file.
    if (m_EnableLayerDetails)
    {
        fs::path filename = m_ModelPath;
        filename.replace_extension("dot");
        delegateOptions.SetSerializeToDot(filename);
    }

    return delegateOptions;
}
#endif
