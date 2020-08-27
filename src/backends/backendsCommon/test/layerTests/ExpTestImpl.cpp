//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ExpTestImpl.hpp"
#include "ElementwiseUnaryTestImpl.hpp"

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Exp2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 2, 2 };

    std::vector<float> inputValues
    {
        3.0f, 2.0f,
        1.0f, 1.1f
    };

    std::vector<float> expectedOutputValues
    {
        20.085536923188f, 7.389056098931f,
         2.718281828459f, 3.004166023946f
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Exp,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> Exp3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 3, 1, 2 };

    std::vector<float> inputValues
    {
        5.0f, 4.0f,
        3.0f, 2.0f,
        1.0f, 1.1f
    };

    std::vector<float> expectedOutputValues
    {
        148.413159102577f, 54.598150033144f,
         20.085536923188f,  7.389056098931f,
          2.718281828459f,  3.004166023946f
    };

    return ElementwiseUnaryTestHelper<3, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Exp,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ExpZeroTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 1, 2 };

    std::vector<float> inputValues
    {
        0.f, 0.f
    };

    std::vector<float> expectedOutputValues
    {
        1.0f, 1.0f
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Exp,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ExpNegativeTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 1, 2 };

    std::vector<float> inputValues
    {
        -5.9f, -5.8f
    };

    std::vector<float> expectedOutputValues
    {
        0.0027394448187683684f, 0.0030275547453758153f,
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Exp,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
Exp2dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
Exp2dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
Exp2dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
Exp2dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
Exp2dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
Exp3dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
Exp3dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
Exp3dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
Exp3dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
Exp3dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ExpZeroTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ExpNegativeTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);