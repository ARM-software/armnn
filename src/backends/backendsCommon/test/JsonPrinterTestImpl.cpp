//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "JsonPrinterTestImpl.hpp"

#include <Profiling.hpp>

#include <armnn/Descriptors.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/unit_test.hpp>

#include <sstream>
#include <stack>
#include <string>

inline bool AreMatchingPair(const char opening, const char closing)
{
    return (opening == '{' && closing == '}') || (opening == '[' && closing == ']');
}

bool AreParenthesesMatching(const std::string& exp)
{
    std::stack<char> expStack;
    for (size_t i = 0; i < exp.length(); ++i)
    {
        if (exp[i] == '{' || exp[i] == '[')
        {
            expStack.push(exp[i]);
        }
        else if (exp[i] == '}' || exp[i] == ']')
        {
            if (expStack.empty() || !AreMatchingPair(expStack.top(), exp[i]))
            {
                return false;
            }
            else
            {
                expStack.pop();
            }
        }
    }
    return expStack.empty();
}

std::vector<double> ExtractMeasurements(const std::string& exp)
{
    std::vector<double> numbers;
    bool inArray = false;
    std::string numberString;
    for (size_t i = 0; i < exp.size(); ++i)
    {
        if (exp[i] == '[')
        {
            inArray = true;
        }
        else if (exp[i] == ']' && inArray)
        {
            try
            {
                boost::trim_if(numberString, boost::is_any_of("\t,\n"));
                numbers.push_back(std::stod(numberString));
            }
            catch (std::invalid_argument const& e)
            {
                BOOST_FAIL("Could not convert measurements to double: " + numberString);
            }

            numberString.clear();
            inArray = false;
        }
        else if (exp[i] == ',' && inArray)
        {
            try
            {
                boost::trim_if(numberString, boost::is_any_of("\t,\n"));
                numbers.push_back(std::stod(numberString));
            }
            catch (std::invalid_argument const& e)
            {
                BOOST_FAIL("Could not convert measurements to double: " + numberString);
            }
            numberString.clear();
        }
        else if (exp[i] != '[' && inArray && exp[i] != ',' && exp[i] != ' ')
        {
            numberString += exp[i];
        }
    }
    return numbers;
}

std::vector<std::string> ExtractSections(const std::string& exp)
{
    std::vector<std::string> sections;

    std::stack<size_t> s;
    for (size_t i = 0; i < exp.size(); i++)
    {
        if (exp.at(i) == '{')
        {
            s.push(i);
        }
        else if (exp.at(i) == '}')
        {
            size_t from = s.top();
            s.pop();
            sections.push_back(exp.substr(from, i - from + 1));
        }
    }

    return sections;
}

std::string GetSoftmaxProfilerJson(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    BOOST_CHECK(!backends.empty());

    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    options.m_EnableGpuProfiling = backends.front() == armnn::Compute::GpuAcc;
    IRuntimePtr runtime(IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0, "input");
    SoftmaxDescriptor softmaxDescriptor;
    // Set Axis to 1 if CL or Neon until further Axes are supported.
    if ( backends.front() == armnn::Compute::CpuAcc || backends.front() == armnn::Compute::GpuAcc)
    {
        softmaxDescriptor.m_Axis = 1;
    }
    IConnectableLayer* softmax = net->AddSoftmaxLayer(softmaxDescriptor, "softmax");
    IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(softmax->GetInputSlot(0));
    softmax->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // set the tensors in the network
    TensorInfo inputTensorInfo(TensorShape({1, 5}), DataType::QuantisedAsymm8);
    inputTensorInfo.SetQuantizationOffset(100);
    inputTensorInfo.SetQuantizationScale(10000.0f);
    input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    TensorInfo outputTensorInfo(TensorShape({1, 5}), DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(1.0f / 256.0f);
    softmax->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    if(!optNet)
    {
        BOOST_FAIL("Error occurred during Optimization, Optimize() returned nullptr.");
    }
    // load it into the runtime
    NetworkId netId;
    auto error = runtime->LoadNetwork(netId, std::move(optNet));
    BOOST_TEST(error == Status::Success);

    // create structures for input & output
    std::vector<uint8_t> inputData
        {
            1, 10, 3, 200, 5
            // one of inputs is sufficiently larger than the others to saturate softmax
        };
    std::vector<uint8_t> outputData(5);

    armnn::InputTensors inputTensors
        {
            {0, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())}
        };
    armnn::OutputTensors outputTensors
        {
            {0, armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
        };

    runtime->GetProfiler(netId)->EnableProfiling(true);

    // do the inferences
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // retrieve the Profiler.Print() output
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);

    return ss.str();
}

inline void ValidateProfilerJson(std::string& result)
{
    // ensure all measurements are greater than zero
    std::vector<double> measurementsVector = ExtractMeasurements(result);
    BOOST_CHECK(!measurementsVector.empty());

    // check sections contain raw and unit tags
    // first ensure Parenthesis are balanced
    if (AreParenthesesMatching(result))
    {
        // remove parent sections that will not have raw or unit tag
        std::vector<std::string> sectionVector = ExtractSections(result);
        for (size_t i = 0; i < sectionVector.size(); ++i)
        {
            if (boost::contains(sectionVector[i], "\"ArmNN\":")
                || boost::contains(sectionVector[i], "\"inference_measurements\":"))
            {
                sectionVector.erase(sectionVector.begin() + static_cast<int>(i));
            }
        }
        BOOST_CHECK(!sectionVector.empty());

        BOOST_CHECK(std::all_of(sectionVector.begin(), sectionVector.end(),
                                [](std::string i) { return boost::contains(i, "\"raw\":"); }));

        BOOST_CHECK(std::all_of(sectionVector.begin(), sectionVector.end(),
                                [](std::string i) { return boost::contains(i, "\"unit\":"); }));
    }

    // remove the time measurements as they vary from test to test
    result.erase(std::remove_if (result.begin(),result.end(),
                                 [](char c) { return c == '.'; }), result.end());
    result.erase(std::remove_if (result.begin(), result.end(), &isdigit), result.end());
    result.erase(std::remove_if (result.begin(),result.end(),
                                 [](char c) { return c == '\t'; }), result.end());

    BOOST_CHECK(boost::contains(result, "ArmNN"));
    BOOST_CHECK(boost::contains(result, "inference_measurements"));

    // ensure no spare parenthesis present in print output
    BOOST_CHECK(AreParenthesesMatching(result));
}

void RunSoftmaxProfilerJsonPrinterTest(const std::vector<armnn::BackendId>& backends)
{
    // setup the test fixture and obtain JSON Printer result
    std::string result = GetSoftmaxProfilerJson(backends);

    // validate the JSON Printer result
    ValidateProfilerJson(result);

    const armnn::BackendId& firstBackend = backends.at(0);
    if (firstBackend == armnn::Compute::GpuAcc)
    {
        BOOST_CHECK(boost::contains(result,
            "OpenClKernelTimer/: softmax_layer_max_shift_exp_sum_quantized_serial GWS[,,]"));
    }
    else if (firstBackend == armnn::Compute::CpuAcc)
    {
        BOOST_CHECK(boost::contains(result,
            "NeonKernelTimer/: NEFillBorderKernel"));
    }
}
