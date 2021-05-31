//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LogTestImpl.hpp"
#include "ElementwiseUnaryTestImpl.hpp"

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Log2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 2, 2 };

    std::vector<float> inputValues
    {
        3.0f, 2.7182818284f,
        1.0f, 1.1f
    };

    std::vector<float> expectedOutputValues
    {
        1.09861228867f, 0.99999999997f,
        0.0f, 0.0953101798f
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Log,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> Log3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 3, 1, 2 };

    std::vector<float> inputValues
    {
        5.0f, 4.0f,
        3.0f, 2.7182818284f,
        1.0f, 1.1f
    };

    std::vector<float> expectedOutputValues
    {
        1.60943791243f, 1.38629436112f,
        1.09861228867f, 0.99999999997f,
        0.0f, 0.0953101798f
    };

    return ElementwiseUnaryTestHelper<3, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Log,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> LogZeroTest(
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
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Log,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> LogNegativeTest(
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
        -std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN()
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Log,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

//
// Loglicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
Log2dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
Log2dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
Log2dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
Log2dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
Log2dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
Log3dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
Log3dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
Log3dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
Log3dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
Log3dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
LogZeroTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
LogNegativeTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);