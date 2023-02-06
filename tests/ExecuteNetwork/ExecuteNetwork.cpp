//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ExecuteNetworkProgramOptions.hpp"
#include "ArmNNExecutor.hpp"
#if defined(ARMNN_TFLITE_DELEGATE)
#include "TfliteExecutor.hpp"
#endif
#include <armnn/Logging.hpp>


std::unique_ptr<IExecutor> BuildExecutor(ProgramOptions& programOptions)
{
    if (programOptions.m_ExNetParams.m_TfLiteExecutor == ExecuteNetworkParams::TfLiteExecutor::ArmNNTfLiteDelegate ||
        programOptions.m_ExNetParams.m_TfLiteExecutor == ExecuteNetworkParams::TfLiteExecutor::TfliteInterpreter)
    {
#if defined(ARMNN_TFLITE_DELEGATE)
        return std::make_unique<TfLiteExecutor>(programOptions.m_ExNetParams, programOptions.m_RuntimeOptions);
#else
        ARMNN_LOG(fatal) << "Not built with Arm NN Tensorflow-Lite delegate support.";
        return nullptr;
#endif
    }
    else
    {
        return std::make_unique<ArmNNExecutor>(programOptions.m_ExNetParams, programOptions.m_RuntimeOptions);
    }
}

// MAIN
int main(int argc, const char* argv[])
{
    // Configures logging for both the ARMNN library and this test program.
#ifdef NDEBUG
    armnn::LogSeverity level = armnn::LogSeverity::Info;
#else
    armnn::LogSeverity level = armnn::LogSeverity::Debug;
#endif
    armnn::ConfigureLogging(true, true, level);


    // Get ExecuteNetwork parameters and runtime options from command line
    // This might throw an InvalidArgumentException if the user provided invalid inputs
    ProgramOptions programOptions;
    try
    {
        programOptions.ParseOptions(argc, argv);
    }
    catch (const std::exception& e)
    {
        ARMNN_LOG(fatal) << e.what();
        return EXIT_FAILURE;
    }

    std::vector<const void*> outputResults;
    std::unique_ptr<IExecutor> executor;
    try
    {
        executor = BuildExecutor(programOptions);
        if ((!executor) || (executor->m_constructionFailed))
        {
            return EXIT_FAILURE;
        }
    }
    catch (const std::exception& e)
    {
        ARMNN_LOG(fatal) << e.what();
        return EXIT_FAILURE;
    }


    executor->PrintNetworkInfo();
    outputResults = executor->Execute();

    if (!programOptions.m_ExNetParams.m_ComparisonComputeDevices.empty() ||
         programOptions.m_ExNetParams.m_CompareWithTflite)
    {
        ExecuteNetworkParams comparisonParams = programOptions.m_ExNetParams;
        comparisonParams.m_ComputeDevices = programOptions.m_ExNetParams.m_ComparisonComputeDevices;

        if (programOptions.m_ExNetParams.m_CompareWithTflite)
        {
            comparisonParams.m_TfLiteExecutor = ExecuteNetworkParams::TfLiteExecutor::TfliteInterpreter;
        }

        auto comparisonExecutor = BuildExecutor(programOptions);

        if (!comparisonExecutor)
        {
            return EXIT_FAILURE;
        }

        comparisonExecutor->PrintNetworkInfo();
        comparisonExecutor->Execute();

        comparisonExecutor->CompareAndPrintResult(outputResults);
    }
}
