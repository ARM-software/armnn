//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>

#include <armnn/Utils.hpp>
#include <armnn/Types.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/Descriptors.hpp>

BOOST_AUTO_TEST_SUITE(Utils)

BOOST_AUTO_TEST_CASE(DataTypeSize)
{
    BOOST_TEST(armnn::GetDataTypeSize(armnn::DataType::Float32) == 4);
    BOOST_TEST(armnn::GetDataTypeSize(armnn::DataType::QuantisedAsymm8) == 1);
    BOOST_TEST(armnn::GetDataTypeSize(armnn::DataType::Signed32) == 4);
}

BOOST_AUTO_TEST_CASE(GetDataTypeTest)
{
    BOOST_TEST((armnn::GetDataType<float>() == armnn::DataType::Float32));
    BOOST_TEST((armnn::GetDataType<uint8_t>() == armnn::DataType::QuantisedAsymm8));
    BOOST_TEST((armnn::GetDataType<int32_t>() == armnn::DataType::Signed32));
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithTooManyMappings)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 0u, 1u, 2u, 3u, 4u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithInvalidMappings1d)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 1u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithInvalidMappings2d)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 2u, 0u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithInvalidMappings3d)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 0u, 3u, 1u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithInvalidMappings4d)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 0u, 1u, 2u, 4u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithDuplicatedMappings)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 1u, 1u, 0u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_SUITE_END()
