//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "arm-armnn-sl"

#include "DriverOptions.hpp"

#include "CanonicalUtils.hpp"

#include <armnn/Version.hpp>
#include <log/log.h>
#include "SystemPropertiesUtils.hpp"

#include <OperationsUtils.h>

#include <cxxopts/cxxopts.hpp>

#include <algorithm>
#include <cassert>
#include <functional>
#include <string>
#include <sstream>

using namespace android;
using namespace std;

namespace armnn_driver
{

DriverOptions::DriverOptions(armnn::Compute computeDevice, bool fp16Enabled)
    : m_Backends({computeDevice})
    , m_VerboseLogging(false)
    , m_RequestInputsAndOutputsDumpDir(std::string(""))
    , m_ServiceName(std::string("armnn_sl"))
    , m_ForcedUnsupportedOperations({})
    , m_ClTunedParametersMode(armnn::IGpuAccTunedParameters::Mode::UseTunedParameters)
    , m_ClTuningLevel(armnn::IGpuAccTunedParameters::TuningLevel::Rapid)
    , m_EnableGpuProfiling(false)
    , m_fp16Enabled(fp16Enabled)
    , m_FastMathEnabled(false)
    , m_ShouldExit(false)
    , m_ExitCode(EXIT_SUCCESS)
    , m_CachedNetworkFilePath(std::string(""))
    , m_SaveCachedNetwork(false)
    , m_NumberOfThreads(0)
    , m_EnableAsyncModelExecution(false)
    , m_ArmnnNumberOfThreads(1)
{
}

DriverOptions::DriverOptions(const std::vector<armnn::BackendId>& backends, bool fp16Enabled)
    : m_Backends(backends)
    , m_VerboseLogging(false)
    , m_RequestInputsAndOutputsDumpDir(std::string(""))
    , m_ServiceName(std::string("armnn_sl"))
    , m_ForcedUnsupportedOperations({})
    , m_ClTunedParametersMode(armnn::IGpuAccTunedParameters::Mode::UseTunedParameters)
    , m_ClTuningLevel(armnn::IGpuAccTunedParameters::TuningLevel::Rapid)
    , m_EnableGpuProfiling(false)
    , m_fp16Enabled(fp16Enabled)
    , m_FastMathEnabled(false)
    , m_ShouldExit(false)
    , m_ExitCode(EXIT_SUCCESS)
    , m_CachedNetworkFilePath(std::string(""))
    , m_SaveCachedNetwork(false)
    , m_NumberOfThreads(0)
    , m_EnableAsyncModelExecution(false)
    , m_ArmnnNumberOfThreads(1)
{
}

// This default constructor will example an environment variable called
// ARMNN_SL_OPTIONS.  It will parse the parameters using the existing cxx
// opts mechanism.
DriverOptions::DriverOptions()
    : m_VerboseLogging(false)
    , m_RequestInputsAndOutputsDumpDir(std::string(""))
    , m_ServiceName(std::string("armnn_sl"))
    , m_ForcedUnsupportedOperations({})
    , m_ClTunedParametersMode(armnn::IGpuAccTunedParameters::Mode::UseTunedParameters)
    , m_ClTuningLevel(armnn::IGpuAccTunedParameters::TuningLevel::Rapid)
    , m_EnableGpuProfiling(false)
    , m_fp16Enabled(false)
    , m_FastMathEnabled(false)
    , m_ShouldExit(false)
    , m_SaveCachedNetwork(false)
    , m_NumberOfThreads(0)
    , m_EnableAsyncModelExecution(false)
    , m_ArmnnNumberOfThreads(1)
{
    std::string unsupportedOperationsAsString;
    std::string clTunedParametersModeAsString;
    std::string clTuningLevelAsString;
    std::vector<std::string> backends;
    bool showHelp = false;
    bool showVersion = false;

    const char* rawEnv = std::getenv("ARMNN_SL_OPTIONS");
    // If the environment variable isn't set we'll continue as if it were an empty string.
    if (!rawEnv)
    {
        rawEnv = "";
    }
    string optionsAsString(rawEnv);
    regex whiteSpaceRegex("\\s+");
    // Tokienize the string based on whitespace.
    sregex_token_iterator iter(optionsAsString.begin(), optionsAsString.end(), whiteSpaceRegex, -1);  
    sregex_token_iterator end;
    vector<string> cliAsVector(iter, end);
    // As we're pretending to be a command line, argv[0] should be an executable name.
    cliAsVector.insert(cliAsVector.begin(), "ARMNN_SL_OPTIONS");
    // Convert the vector of string to a vector of char* backed by the existing vector.
    std::vector<char*> argVector;
    for (const auto& arg : cliAsVector)
    {
        argVector.push_back((char*)arg.data());
    }
    // Terminate the array.
    argVector.push_back(nullptr);
    // Create usable variables.
    int argc = argVector.size() - 1; // Ignore the null pointer at the end.
    char** argv = argVector.data();

    cxxopts::Options optionsDesc(argv[0], "Arm NN Support Library for the Android Neural Networks API."
                                          "The support library will convert Android NNAPI requests "
                                          "and delegate them to available ArmNN backends.");
    try
    {
        optionsDesc.add_options()

        ("a,enable-fast-math",
                "Enables fast_math options in backends that support it. Using the fast_math flag can "
               "lead to performance improvements but may result in reduced or different precision.",
         cxxopts::value<bool>(m_FastMathEnabled)->default_value("false"))

        ("c,compute","Comma separated list of backends to run layers on. "
                "Examples of possible values are: CpuRef, CpuAcc, GpuAcc",
         cxxopts::value<std::vector<std::string>>(backends))

        ("d,request-inputs-and-outputs-dump-dir",
         "If non-empty, the directory where request inputs and outputs should be dumped",
         cxxopts::value<std::string>(m_RequestInputsAndOutputsDumpDir)->default_value(""))

        ("f,fp16-enabled", "Enables support for relaxed computation from Float32 to Float16",
         cxxopts::value<bool>(m_fp16Enabled)->default_value("false"))

        ("h,help", "Show this help",
         cxxopts::value<bool>(showHelp)->default_value("false")->implicit_value("true"))

        ("m,cl-tuned-parameters-mode",
         "If 'UseTunedParameters' (the default), will read CL tuned parameters from the file specified by "
         "--cl-tuned-parameters-file. "
         "If 'UpdateTunedParameters', will also find the optimum parameters when preparing new networks and update "
         "the file accordingly.",
         cxxopts::value<std::string>(clTunedParametersModeAsString)->default_value("UseTunedParameters"))

        ("g,mlgo-cl-tuned-parameters-file",
        "If non-empty, the given file will be used to load/save MLGO CL tuned parameters. ",
        cxxopts::value<std::string>(m_ClMLGOTunedParametersFile)->default_value(""))

        ("o,cl-tuning-level",
         "exhaustive: all lws values are tested "
         "normal: reduced number of lws values but enough to still have the performance really close to the "
         "exhaustive approach "
         "rapid: only 3 lws values should be tested for each kernel ",
         cxxopts::value<std::string>(clTuningLevelAsString)->default_value("rapid"))

        ("p,gpu-profiling", "Turns GPU profiling on",
         cxxopts::value<bool>(m_EnableGpuProfiling)->default_value("false"))

        ("q,cached-network-file", "If non-empty, the given file will be used to load/save cached network. "
                                   "If save-cached-network option is given will save the cached network to given file."
                                   "If save-cached-network option is not given will load the cached network from given "
                                   "file.",
        cxxopts::value<std::string>(m_CachedNetworkFilePath)->default_value(""))

        ("s,save-cached-network",
                "Enables saving the cached network to the file given with cached-network-file option."
                " See also --cached-network-file",
        cxxopts::value<bool>(m_SaveCachedNetwork)->default_value("false"))

        ("number-of-threads",
         "Assign the number of threads used by the CpuAcc backend. "
         "Input value must be between 1 and 64. "
         "Default is set to 0 (Backend will decide number of threads to use).",
         cxxopts::value<unsigned int>(m_NumberOfThreads)->default_value("0"))

        ("t,cl-tuned-parameters-file",
         "If non-empty, the given file will be used to load/save CL tuned parameters. "
         "See also --cl-tuned-parameters-mode",
         cxxopts::value<std::string>(m_ClTunedParametersFile)->default_value(""))

        ("u,unsupported-operations",
         "If non-empty, a comma-separated list of operation indices which the driver will forcibly "
         "consider unsupported",
         cxxopts::value<std::string>(unsupportedOperationsAsString)->default_value(""))

        ("v,verbose-logging", "Turns verbose logging on",
         cxxopts::value<bool>(m_VerboseLogging)->default_value("false")->implicit_value("true"))

        ("V,version", "Show version information",
         cxxopts::value<bool>(showVersion)->default_value("false")->implicit_value("true"))
        ;
    }
    catch (const std::exception& e)
    {
        VLOG(DRIVER) << "An error occurred attempting to construct options: " << e.what();
        std::cout << "An error occurred attempting to construct options: %s" << std::endl;
        m_ExitCode = EXIT_FAILURE;
        return;
    }

    try
    {
        cxxopts::ParseResult result = optionsDesc.parse(argc, argv);
    }
    catch (const cxxopts::OptionException& e)
    {
        VLOG(DRIVER) << "An exception occurred attempting to parse program options: " << e.what();
        std::cout << optionsDesc.help() << std::endl
                  << "An exception occurred while parsing program options: " << std::endl
                  << e.what() << std::endl;
        m_ShouldExit = true;
        m_ExitCode = EXIT_FAILURE;
        return;
    }
    if (showHelp)
    {
        VLOG(DRIVER) << "Showing help and exiting";
        std::cout << optionsDesc.help() << std::endl;
        m_ShouldExit = true;
        m_ExitCode = EXIT_SUCCESS;
        return;
    }
    if (showVersion)
    {
        VLOG(DRIVER) << "Showing version and exiting";
        std::cout << "ArmNN Android NN driver for the Android Neural Networks API.\n"
                     "ArmNN v" << ARMNN_VERSION << std::endl;
        m_ShouldExit = true;
        m_ExitCode = EXIT_SUCCESS;
        return;
    }

    // Convert the string backend names into backendId's.
    m_Backends.reserve(backends.size());
    for (auto&& backend : backends)
    {
        m_Backends.emplace_back(backend);
    }

    // If no backends have been specified then the default value is GpuAcc.
    if (backends.empty())
    {
        VLOG(DRIVER) << "No backends have been specified:";
        std::cout << optionsDesc.help() << std::endl
                  << "Unable to start:" << std::endl
                  << "No backends have been specified" << std::endl;
        m_ShouldExit = true;
        m_ExitCode = EXIT_FAILURE;
        return;
    }

    if (!unsupportedOperationsAsString.empty())
    {
        std::istringstream argStream(unsupportedOperationsAsString);

        std::string s;
        while (!argStream.eof())
        {
            std::getline(argStream, s, ',');
            try
            {
                unsigned int operationIdx = std::stoi(s);
                m_ForcedUnsupportedOperations.insert(operationIdx);
            }
            catch (const std::invalid_argument&)
            {
                VLOG(DRIVER) << "Ignoring invalid integer argument in -u/--unsupported-operations value: " << s.c_str();
            }
        }
    }

    if (!m_ClTunedParametersFile.empty())
    {
        // The mode is only relevant if the file path has been provided
        if (clTunedParametersModeAsString == "UseTunedParameters")
        {
            m_ClTunedParametersMode = armnn::IGpuAccTunedParameters::Mode::UseTunedParameters;
        }
        else if (clTunedParametersModeAsString == "UpdateTunedParameters")
        {
            m_ClTunedParametersMode = armnn::IGpuAccTunedParameters::Mode::UpdateTunedParameters;
        }
        else
        {
            VLOG(DRIVER) << "Requested unknown cl-tuned-parameters-mode "
                          << clTunedParametersModeAsString.c_str() << ". Defaulting to UseTunedParameters";
        }

        if (clTuningLevelAsString == "exhaustive")
        {
            m_ClTuningLevel = armnn::IGpuAccTunedParameters::TuningLevel::Exhaustive;
        }
        else if (clTuningLevelAsString == "normal")
        {
            m_ClTuningLevel = armnn::IGpuAccTunedParameters::TuningLevel::Normal;
        }
        else if (clTuningLevelAsString == "rapid")
        {
            m_ClTuningLevel = armnn::IGpuAccTunedParameters::TuningLevel::Rapid;
        }
        else
        {
            VLOG(DRIVER) << "Requested unknown cl-tuner-mode '%s'. "
                            "Defaulting to rapid" << clTuningLevelAsString.c_str();
        }
    }
}

} // namespace armnn_driver
