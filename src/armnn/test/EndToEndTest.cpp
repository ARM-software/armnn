//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Descriptors.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>

#include <backendsCommon/test/QuantizeHelper.hpp>

#include <boost/core/ignore_unused.hpp>
#include <boost/test/unit_test.hpp>

#include <set>

BOOST_AUTO_TEST_SUITE(EndToEnd)

namespace
{

template<typename T>
bool IsFloatIterFunc(T iter)
{
    boost::ignore_unused(iter);
    return IsFloatingPointIterator<T>::value;
}

} //namespace

BOOST_AUTO_TEST_CASE(QuantizedHelper)
{
    std::vector<float> fArray;
    BOOST_TEST(IsFloatIterFunc(fArray.begin()) == true);
    BOOST_TEST(IsFloatIterFunc(fArray.cbegin()) == true);

    std::vector<double> dArray;
    BOOST_TEST(IsFloatIterFunc(dArray.begin()) == true);

    std::vector<int> iArray;
    BOOST_TEST(IsFloatIterFunc(iArray.begin()) == false);

    float floats[5];
    BOOST_TEST(IsFloatIterFunc(&floats[0]) == true);

    int ints[5];
    BOOST_TEST(IsFloatIterFunc(&ints[0]) == false);
}

BOOST_AUTO_TEST_CASE(ErrorOnLoadNetwork)
{
    using namespace armnn;

    // Create runtime in which test will run
    // Note we don't allow falling back to CpuRef if an operation (excluding inputs, outputs, etc.) isn't supported
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc and isn't allowed to fall back, so Optimize will return null.
    NormalizationDescriptor descriptor;
    IConnectableLayer* pooling = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // optimize the network
    std::vector<BackendId> backends = {Compute::CpuAcc};
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(!optNet);
}

BOOST_AUTO_TEST_SUITE_END()
