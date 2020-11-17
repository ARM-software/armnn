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
    else if (modelFormat.find("caffe") != std::string::npos)
    {
#if defined(ARMNN_CAFFE_PARSER)
#else
        throw armnn::InvalidArgumentException("Can't run model in caffe format without a "
                                              "built with Caffe parser support.");
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
    else if (modelFormat.find("tensorflow") != std::string::npos)
    {
#if defined(ARMNN_TF_PARSER)
#else
        throw armnn::InvalidArgumentException("Can't run model in onnx format without a "
                                              "built with Tensorflow parser support.");
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
                                                          "Please include 'caffe', 'tensorflow', 'tflite' or 'onnx'",
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
    // Set to true if it is preferred to throw an exception rather than use ARMNN_LOG
    bool throwExc = false;

    try
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
            ARMNN_LOG(fatal) << "BFloat16 and Float16 turbo mode cannot be enabled at the same time.";
        }

        m_IsModelBinary = IsModelBinary(m_ModelFormat);

        CheckModelFormat(m_ModelFormat);

        // Check input tensor shapes
        if ((m_InputTensorShapes.size() != 0) &&
            (m_InputTensorShapes.size() != m_InputNames.size()))
        {
            ARMNN_LOG(fatal) << "input-name and input-tensor-shape must have the same amount of elements. ";
        }

        if (m_InputTensorDataFilePaths.size() != 0)
        {
            if (!ValidatePaths(m_InputTensorDataFilePaths, true))
            {
                ARMNN_LOG(fatal) << "One or more input data file paths are not valid. ";
            }

            if (m_InputTensorDataFilePaths.size() != m_InputNames.size())
            {
                ARMNN_LOG(fatal) << "input-name and input-tensor-data must have the same amount of elements. ";
            }
        }

        if ((m_OutputTensorFiles.size() != 0) &&
            (m_OutputTensorFiles.size() != m_OutputNames.size()))
        {
            ARMNN_LOG(fatal) << "output-name and write-outputs-to-file must have the same amount of elements. ";
        }

        if (m_InputTypes.size() == 0)
        {
            //Defaults the value of all inputs to "float"
            m_InputTypes.assign(m_InputNames.size(), "float");
        }
        else if ((m_InputTypes.size() != 0) &&
                 (m_InputTypes.size() != m_InputNames.size()))
        {
            ARMNN_LOG(fatal) << "input-name and input-type must have the same amount of elements.";
        }

        if (m_OutputTypes.size() == 0)
        {
            //Defaults the value of all outputs to "float"
            m_OutputTypes.assign(m_OutputNames.size(), "float");
        }
        else if ((m_OutputTypes.size() != 0) &&
                 (m_OutputTypes.size() != m_OutputNames.size()))
        {
            ARMNN_LOG(fatal) << "output-name and output-type must have the same amount of elements.";
        }

        // Check that threshold time is not less than zero
        if (m_ThresholdTime < 0)
        {
            ARMNN_LOG(fatal) << "Threshold time supplied as a command line argument is less than zero.";
        }
    }
    catch (std::string& exc)
    {
        if (throwExc)
        {
            throw armnn::InvalidArgumentException(exc);
        }
        else
        {
            std::cout << exc;
            exit(EXIT_FAILURE);
        }
    }
    // Check turbo modes

    // Warn if ExecuteNetwork will generate dummy input data
    if (m_GenerateTensorData)
    {
        ARMNN_LOG(warning) << "No input files provided, input tensors will be filled with 0s.";
    }
}