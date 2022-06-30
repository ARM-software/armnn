//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Exceptions.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/Workload.hpp>

#include <doctest/doctest.h>

#include <thread>

using namespace armnn;


namespace
{

TEST_SUITE("WorkloadAsyncExecuteTests")
{

struct Workload0 : BaseWorkload<ElementwiseUnaryQueueDescriptor>
{
    Workload0(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload(descriptor, info)
    {
    }

    Workload0() : BaseWorkload(ElementwiseUnaryQueueDescriptor(), WorkloadInfo())
    {
    }

    void Execute() const
    {
        int* inVals = static_cast<int*>(m_Data.m_Inputs[0][0].Map());
        int* outVals = static_cast<int*>(m_Data.m_Outputs[0][0].Map());

        for (unsigned int i = 0;
             i < m_Data.m_Inputs[0][0].GetShape().GetNumElements();
             ++i)
        {
            outVals[i] = inVals[i] * outVals[i];
            inVals[i] = outVals[i];
        }
    }

    void ExecuteAsync(ExecutionData& executionData)
    {
        WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
        int* inVals = static_cast<int*>(workingMemDescriptor->m_Inputs[0][0].Map());
        int* outVals = static_cast<int*>(workingMemDescriptor->m_Outputs[0][0].Map());

        for (unsigned int i = 0;
             i < workingMemDescriptor->m_Inputs[0][0].GetShape().GetNumElements();
             ++i)
        {
            outVals[i] = inVals[i] + outVals[i];
            inVals[i] = outVals[i];
        }
    }

    QueueDescriptor* GetQueueDescriptor()
    {
        return &m_Data;
    }
};

struct Workload1 : BaseWorkload<ElementwiseUnaryQueueDescriptor>
{
    Workload1(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload(descriptor, info)
    {
    }

    void Execute() const
    {
        int* inVals = static_cast<int*>(m_Data.m_Inputs[0][0].Map());
        int* outVals = static_cast<int*>(m_Data.m_Outputs[0][0].Map());

        for (unsigned int i = 0;
             i < m_Data.m_Inputs[0][0].GetShape().GetNumElements();
             ++i)
        {
            outVals[i] = inVals[i] * outVals[i];
            inVals[i] = outVals[i];
        }
    }
};

void ValidateTensor(ITensorHandle* tensorHandle, int expectedValue)
{
    int* actualOutput = static_cast<int*>(tensorHandle->Map());

    bool allValuesCorrect = true;
    for (unsigned int i = 0;
         i < tensorHandle->GetShape().GetNumElements();
         ++i)
    {
        if (actualOutput[i] != expectedValue)
        {
            allValuesCorrect = false;
        }
    }

    CHECK(allValuesCorrect);
}

template<typename Workload>
std::unique_ptr<Workload> CreateWorkload(TensorInfo info, ITensorHandle* inputTensor, ITensorHandle* outputTensor)
{
    WorkloadInfo workloadInfo;
    workloadInfo.m_InputTensorInfos = std::vector<TensorInfo>{info};
    workloadInfo.m_OutputTensorInfos = std::vector<TensorInfo>{info};

    ElementwiseUnaryQueueDescriptor elementwiseUnaryQueueDescriptor;
    elementwiseUnaryQueueDescriptor.m_Inputs = std::vector<ITensorHandle*>{inputTensor};
    elementwiseUnaryQueueDescriptor.m_Outputs = std::vector<ITensorHandle*>{outputTensor};

    return std::make_unique<Workload>(elementwiseUnaryQueueDescriptor, workloadInfo);
}

TEST_CASE("TestAsyncExecute")
{
    TensorInfo info({5}, DataType::Signed32, 0.0, 0, true);

    int inVals[5]{2, 2, 2, 2, 2};
    int outVals[5]{1, 1, 1, 1, 1};

    int expectedExecuteval = 2;
    int expectedExecuteAsyncval = 3;

    ConstTensor constInputTensor(info, inVals);
    ConstTensor constOutputTensor(info, outVals);

    ScopedTensorHandle syncInput0(constInputTensor);
    ScopedTensorHandle syncOutput0(constOutputTensor);

    std::unique_ptr<Workload0> workload0 = CreateWorkload<Workload0>(info, &syncInput0, &syncOutput0);

    workload0.get()->Execute();

    ScopedTensorHandle asyncInput0(constInputTensor);
    ScopedTensorHandle asyncOutput0(constOutputTensor);

    WorkingMemDescriptor workingMemDescriptor0;
    workingMemDescriptor0.m_Inputs = std::vector<ITensorHandle*>{&asyncInput0};
    workingMemDescriptor0.m_Outputs = std::vector<ITensorHandle*>{&asyncOutput0};

    ExecutionData executionData;
    executionData.m_Data = &workingMemDescriptor0;

    workload0.get()->ExecuteAsync(executionData);

    // Inputs are also changed by the execute/executeAsync calls to make sure there is no interference with them
    ValidateTensor(workingMemDescriptor0.m_Outputs[0], expectedExecuteAsyncval);
    ValidateTensor(workingMemDescriptor0.m_Inputs[0], expectedExecuteAsyncval);

    ValidateTensor(&workload0.get()->GetQueueDescriptor()->m_Outputs[0][0], expectedExecuteval);
    ValidateTensor(&workload0.get()->GetQueueDescriptor()->m_Inputs[0][0], expectedExecuteval);
}

TEST_CASE("TestDefaultAsyncExecute")
{
    TensorInfo info({5}, DataType::Signed32, 0.0f, 0, true);

    std::vector<int> inVals{2, 2, 2, 2, 2};
    std::vector<int> outVals{1, 1, 1, 1, 1};
    std::vector<int> defaultVals{0, 0, 0, 0, 0};

    int expectedExecuteval = 2;

    ConstTensor constInputTensor(info, inVals);
    ConstTensor constOutputTensor(info, outVals);
    ConstTensor defaultTensor(info, &defaultVals);

    ScopedTensorHandle defaultInput = ScopedTensorHandle(defaultTensor);
    ScopedTensorHandle defaultOutput = ScopedTensorHandle(defaultTensor);

    std::unique_ptr<Workload1> workload1 = CreateWorkload<Workload1>(info, &defaultInput, &defaultOutput);

    ScopedTensorHandle asyncInput(constInputTensor);
    ScopedTensorHandle asyncOutput(constOutputTensor);

    WorkingMemDescriptor workingMemDescriptor;
    workingMemDescriptor.m_Inputs = std::vector<ITensorHandle*>{&asyncInput};
    workingMemDescriptor.m_Outputs = std::vector<ITensorHandle*>{&asyncOutput};

    ExecutionData executionData;
    executionData.m_Data = &workingMemDescriptor;

    workload1.get()->ExecuteAsync(executionData);

    // workload1 has no AsyncExecute implementation and so should use the default workload AsyncExecute
    // implementation which will call  workload1.Execute() in a thread safe manner
    ValidateTensor(workingMemDescriptor.m_Outputs[0], expectedExecuteval);
    ValidateTensor(workingMemDescriptor.m_Inputs[0], expectedExecuteval);
}

TEST_CASE("TestDefaultAsyncExeuteWithThreads")
{
    // Use a large vector so the threads have a chance to interact
    unsigned int vecSize = 1000;
    TensorInfo info({vecSize}, DataType::Signed32, 0.0f, 0, true);

    std::vector<int> inVals1(vecSize, 2);
    std::vector<int> outVals1(vecSize, 1);
    std::vector<int> inVals2(vecSize, 5);
    std::vector<int> outVals2(vecSize, -1);

    std::vector<int> defaultVals(vecSize, 0);

    int expectedExecuteval1 = 4;
    int expectedExecuteval2 = 25;
    ConstTensor constInputTensor1(info, inVals1);
    ConstTensor constOutputTensor1(info, outVals1);

    ConstTensor constInputTensor2(info, inVals2);
    ConstTensor constOutputTensor2(info, outVals2);

    ConstTensor defaultTensor(info, defaultVals.data());

    ScopedTensorHandle defaultInput = ScopedTensorHandle(defaultTensor);
    ScopedTensorHandle defaultOutput = ScopedTensorHandle(defaultTensor);
    std::unique_ptr<Workload1> workload = CreateWorkload<Workload1>(info, &defaultInput, &defaultOutput);

    ScopedTensorHandle asyncInput1(constInputTensor1);
    ScopedTensorHandle asyncOutput1(constOutputTensor1);

    WorkingMemDescriptor workingMemDescriptor1;
    workingMemDescriptor1.m_Inputs = std::vector<ITensorHandle*>{&asyncInput1};
    workingMemDescriptor1.m_Outputs = std::vector<ITensorHandle*>{&asyncOutput1};

    ExecutionData executionData1;
    executionData1.m_Data = &workingMemDescriptor1;

    ScopedTensorHandle asyncInput2(constInputTensor2);
    ScopedTensorHandle asyncOutput2(constOutputTensor2);

    WorkingMemDescriptor workingMemDescriptor2;
    workingMemDescriptor2.m_Inputs = std::vector<ITensorHandle*>{&asyncInput2};
    workingMemDescriptor2.m_Outputs = std::vector<ITensorHandle*>{&asyncOutput2};

    ExecutionData executionData2;
    executionData2.m_Data = &workingMemDescriptor2;

    std::thread thread1 = std::thread([&]()
                                      {
                                          workload.get()->ExecuteAsync(executionData1);
                                          workload.get()->ExecuteAsync(executionData1);
                                      });

    std::thread thread2 = std::thread([&]()
                                      {
                                          workload.get()->ExecuteAsync(executionData2);
                                          workload.get()->ExecuteAsync(executionData2);
                                      });

    thread1.join();
    thread2.join();

    ValidateTensor(workingMemDescriptor1.m_Outputs[0], expectedExecuteval1);
    ValidateTensor(workingMemDescriptor1.m_Inputs[0], expectedExecuteval1);

    ValidateTensor(workingMemDescriptor2.m_Outputs[0], expectedExecuteval2);
    ValidateTensor(workingMemDescriptor2.m_Inputs[0], expectedExecuteval2);
}

}

}
