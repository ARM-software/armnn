//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

// Adds unit test framework for interpreting TensorInfo type.
std::ostream& boost_test_print_type(std::ostream& ostr, const TensorInfo& right)
{
    ostr << "TensorInfo[ "
    << right.GetNumDimensions() << ","
    << right.GetShape()[0] << ","
    << right.GetShape()[1] << ","
    << right.GetShape()[2] << ","
    << right.GetShape()[3]
    << " ]" << std::endl;
    return ostr;
}

std::ostream& boost_test_print_type(std::ostream& ostr, const TensorShape& shape)
{
    ostr << "TensorShape[ "
        << shape.GetNumDimensions() << ","
        << shape[0] << ","
        << shape[1] << ","
        << shape[2] << ","
        << shape[3]
        << " ]" << std::endl;
    return ostr;
}

} //namespace armnn
using namespace armnn;

BOOST_AUTO_TEST_SUITE(Tensor)

struct TensorInfoFixture
{
    TensorInfoFixture()
    {
        unsigned int sizes[] = {6,7,8,9};
        m_TensorInfo = TensorInfo(4, sizes, DataType::Float32);
    }
    ~TensorInfoFixture() {};

    TensorInfo m_TensorInfo;
};

BOOST_FIXTURE_TEST_CASE(ConstructShapeUsingListInitialization, TensorInfoFixture)
{
    TensorShape listInitializedShape{ 6, 7, 8, 9 };
    BOOST_TEST(listInitializedShape == m_TensorInfo.GetShape());
}

BOOST_FIXTURE_TEST_CASE(ConstructTensorInfo, TensorInfoFixture)
{
    BOOST_TEST(m_TensorInfo.GetNumDimensions() == 4);
    BOOST_TEST(m_TensorInfo.GetShape()[0] == 6); // <= Outer most
    BOOST_TEST(m_TensorInfo.GetShape()[1] == 7);
    BOOST_TEST(m_TensorInfo.GetShape()[2] == 8);
    BOOST_TEST(m_TensorInfo.GetShape()[3] == 9);     // <= Inner most
}

BOOST_FIXTURE_TEST_CASE(CopyConstructTensorInfo, TensorInfoFixture)
{
    TensorInfo copyConstructed(m_TensorInfo);
    BOOST_TEST(copyConstructed.GetNumDimensions() == 4);
    BOOST_TEST(copyConstructed.GetShape()[0] == 6);
    BOOST_TEST(copyConstructed.GetShape()[1] == 7);
    BOOST_TEST(copyConstructed.GetShape()[2] == 8);
    BOOST_TEST(copyConstructed.GetShape()[3] == 9);
}

BOOST_FIXTURE_TEST_CASE(TensorInfoEquality, TensorInfoFixture)
{
    TensorInfo copyConstructed(m_TensorInfo);
    BOOST_TEST(copyConstructed == m_TensorInfo);
}

BOOST_FIXTURE_TEST_CASE(TensorInfoInequality, TensorInfoFixture)
{
    TensorInfo other;
    unsigned int sizes[] = {2,3,4,5};
    other = TensorInfo(4, sizes, DataType::Float32);

    BOOST_TEST(other != m_TensorInfo);
}

BOOST_FIXTURE_TEST_CASE(TensorInfoAssignmentOperator, TensorInfoFixture)
{
    TensorInfo copy;
    copy = m_TensorInfo;
    BOOST_TEST(copy == m_TensorInfo);
}

void CheckTensor(const ConstTensor& t)
{
    t.GetInfo();
}

BOOST_AUTO_TEST_CASE(TensorVsConstTensor)
{
    int mutableDatum = 2;
    const int immutableDatum = 3;

    armnn::Tensor uninitializedTensor;
    armnn::ConstTensor uninitializedTensor2;

    uninitializedTensor2 = uninitializedTensor;

    armnn::Tensor t(TensorInfo(), &mutableDatum);
    armnn::ConstTensor ct(TensorInfo(), &immutableDatum);

    // Checks that both Tensor and ConstTensor can be passed as a ConstTensor.
    CheckTensor(t);
    CheckTensor(ct);
}

BOOST_AUTO_TEST_CASE(ModifyTensorInfo)
{
    TensorInfo info;
    info.SetShape({ 5, 6, 7, 8 });
    BOOST_TEST((info.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    info.SetDataType(DataType::QuantisedAsymm8);
    BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));
    info.SetQuantizationScale(10.0f);
    BOOST_TEST(info.GetQuantizationScale() == 10.0f);
    info.SetQuantizationOffset(5);
    BOOST_TEST(info.GetQuantizationOffset() == 5);
}

BOOST_AUTO_TEST_CASE(TensorShapeOperatorBrackets)
{
    TensorShape shape({0,1,2,3});
    // Checks version of operator[] which returns an unsigned int.
    BOOST_TEST(shape[2] == 2);
    // Checks the version of operator[] which returns a reference.
    shape[2] = 20;
    BOOST_TEST(shape[2] == 20);
}

BOOST_AUTO_TEST_SUITE_END()
