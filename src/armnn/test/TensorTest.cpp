//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/utility/IgnoreUnused.hpp>


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

BOOST_AUTO_TEST_CASE(CopyNoQuantizationTensorInfo)
{
    TensorInfo infoA;
    infoA.SetShape({ 5, 6, 7, 8 });
    infoA.SetDataType(DataType::QAsymmU8);

    TensorInfo infoB;
    infoB.SetShape({ 5, 6, 7, 8 });
    infoB.SetDataType(DataType::QAsymmU8);
    infoB.SetQuantizationScale(10.0f);
    infoB.SetQuantizationOffset(5);
    infoB.SetQuantizationDim(Optional<unsigned int>(1));

    BOOST_TEST((infoA.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    BOOST_TEST((infoA.GetDataType() == DataType::QAsymmU8));
    BOOST_TEST(infoA.GetQuantizationScale() == 1);
    BOOST_TEST(infoA.GetQuantizationOffset() == 0);
    BOOST_CHECK(!infoA.GetQuantizationDim().has_value());

    BOOST_TEST(infoA != infoB);
    infoA = infoB;
    BOOST_TEST(infoA == infoB);

    BOOST_TEST((infoA.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    BOOST_TEST((infoA.GetDataType() == DataType::QAsymmU8));
    BOOST_TEST(infoA.GetQuantizationScale() == 10.0f);
    BOOST_TEST(infoA.GetQuantizationOffset() == 5);
    BOOST_CHECK(infoA.GetQuantizationDim().value() == 1);
}

BOOST_AUTO_TEST_CASE(CopyDifferentQuantizationTensorInfo)
{
    TensorInfo infoA;
    infoA.SetShape({ 5, 6, 7, 8 });
    infoA.SetDataType(DataType::QAsymmU8);
    infoA.SetQuantizationScale(10.0f);
    infoA.SetQuantizationOffset(5);
    infoA.SetQuantizationDim(Optional<unsigned int>(1));

    TensorInfo infoB;
    infoB.SetShape({ 5, 6, 7, 8 });
    infoB.SetDataType(DataType::QAsymmU8);
    infoB.SetQuantizationScale(11.0f);
    infoB.SetQuantizationOffset(6);
    infoB.SetQuantizationDim(Optional<unsigned int>(2));

    BOOST_TEST((infoA.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    BOOST_TEST((infoA.GetDataType() == DataType::QAsymmU8));
    BOOST_TEST(infoA.GetQuantizationScale() == 10.0f);
    BOOST_TEST(infoA.GetQuantizationOffset() == 5);
    BOOST_CHECK(infoA.GetQuantizationDim().value() == 1);

    BOOST_TEST(infoA != infoB);
    infoA = infoB;
    BOOST_TEST(infoA == infoB);

    BOOST_TEST((infoA.GetShape() == TensorShape({ 5, 6, 7, 8 })));
    BOOST_TEST((infoA.GetDataType() == DataType::QAsymmU8));
    BOOST_TEST(infoA.GetQuantizationScale() == 11.0f);
    BOOST_TEST(infoA.GetQuantizationOffset() == 6);
    BOOST_CHECK(infoA.GetQuantizationDim().value() == 2);
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
    info.SetDataType(DataType::QAsymmU8);
    BOOST_TEST((info.GetDataType() == DataType::QAsymmU8));
    info.SetQuantizationScale(10.0f);
    BOOST_TEST(info.GetQuantizationScale() == 10.0f);
    info.SetQuantizationOffset(5);
    BOOST_TEST(info.GetQuantizationOffset() == 5);
}

BOOST_AUTO_TEST_CASE(TensorShapeOperatorBrackets)
{
    const TensorShape constShape({0,1,2,3});
    TensorShape shape({0,1,2,3});

    // Checks version of operator[] which returns an unsigned int.
    BOOST_TEST(shape[2] == 2);
    shape[2] = 20;
    BOOST_TEST(shape[2] == 20);

    // Checks the version of operator[] which returns a reference.
    BOOST_TEST(constShape[2] == 2);
}

BOOST_AUTO_TEST_CASE(TensorInfoPerAxisQuantization)
{
    // Old constructor
    TensorInfo tensorInfo0({ 1, 1 }, DataType::Float32, 2.0f, 1);
    BOOST_CHECK(!tensorInfo0.HasMultipleQuantizationScales());
    BOOST_CHECK(tensorInfo0.GetQuantizationScale() == 2.0f);
    BOOST_CHECK(tensorInfo0.GetQuantizationOffset() == 1);
    BOOST_CHECK(tensorInfo0.GetQuantizationScales()[0] == 2.0f);
    BOOST_CHECK(!tensorInfo0.GetQuantizationDim().has_value());

    // Set per-axis quantization scales
    std::vector<float> perAxisScales{ 3.0f, 4.0f };
    tensorInfo0.SetQuantizationScales(perAxisScales);
    BOOST_CHECK(tensorInfo0.HasMultipleQuantizationScales());
    BOOST_CHECK(tensorInfo0.GetQuantizationScales() == perAxisScales);

    // Set per-tensor quantization scale
    tensorInfo0.SetQuantizationScale(5.0f);
    BOOST_CHECK(!tensorInfo0.HasMultipleQuantizationScales());
    BOOST_CHECK(tensorInfo0.GetQuantizationScales()[0] == 5.0f);

    // Set quantization offset
    tensorInfo0.SetQuantizationDim(Optional<unsigned int>(1));
    BOOST_CHECK(tensorInfo0.GetQuantizationDim().value() == 1);

    // New constructor
    perAxisScales = { 6.0f, 7.0f };
    TensorInfo tensorInfo1({ 1, 1 }, DataType::Float32, perAxisScales, 1);
    BOOST_CHECK(tensorInfo1.HasMultipleQuantizationScales());
    BOOST_CHECK(tensorInfo1.GetQuantizationOffset() == 0);
    BOOST_CHECK(tensorInfo1.GetQuantizationScales() == perAxisScales);
    BOOST_CHECK(tensorInfo1.GetQuantizationDim().value() == 1);
}

BOOST_AUTO_TEST_CASE(TensorShape_scalar)
{
    float mutableDatum = 3.1416f;

    const armnn::TensorShape shape  (armnn::Dimensionality::Scalar );
    armnn::TensorInfo        info   ( shape, DataType::Float32 );
    const armnn::Tensor      tensor ( info, &mutableDatum );

    BOOST_CHECK(armnn::Dimensionality::Scalar == shape.GetDimensionality());
    float scalarValue = *reinterpret_cast<float*>(tensor.GetMemoryArea());
    BOOST_CHECK_MESSAGE(mutableDatum == scalarValue, "Scalar value is " << scalarValue);

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    BOOST_TEST(shape_equal == shape);
    BOOST_TEST(shape_different != shape);
    BOOST_CHECK_MESSAGE(1 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    BOOST_CHECK_MESSAGE(1 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    BOOST_CHECK(true == shape.GetDimensionSpecificity(0));
    BOOST_CHECK(shape.AreAllDimensionsSpecified());
    BOOST_CHECK(shape.IsAtLeastOneDimensionSpecified());

    BOOST_TEST(1 == shape[0]);
    BOOST_TEST(1 == tensor.GetShape()[0]);
    BOOST_TEST(1 == tensor.GetInfo().GetShape()[0]);
    BOOST_CHECK_THROW( shape[1], InvalidArgumentException );

    float newMutableDatum  = 42.f;
    std::memcpy(tensor.GetMemoryArea(), &newMutableDatum, sizeof(float));
    scalarValue = *reinterpret_cast<float*>(tensor.GetMemoryArea());
    BOOST_CHECK_MESSAGE(newMutableDatum == scalarValue, "Scalar value is " << scalarValue);
}

BOOST_AUTO_TEST_CASE(TensorShape_DynamicTensorType1_unknownNumberDimensions)
{
    float       mutableDatum  = 3.1416f;

    armnn::TensorShape shape  (armnn::Dimensionality::NotSpecified );
    armnn::TensorInfo  info   ( shape, DataType::Float32 );
    armnn::Tensor      tensor ( info, &mutableDatum );

    BOOST_CHECK(armnn::Dimensionality::NotSpecified == shape.GetDimensionality());
    BOOST_CHECK_THROW( shape[0], InvalidArgumentException );
    BOOST_CHECK_THROW( shape.GetNumElements(), InvalidArgumentException );
    BOOST_CHECK_THROW( shape.GetNumDimensions(), InvalidArgumentException );

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    BOOST_TEST(shape_equal == shape);
    BOOST_TEST(shape_different != shape);
}

BOOST_AUTO_TEST_CASE(TensorShape_DynamicTensorType1_unknownAllDimensionsSizes)
{
    float       mutableDatum  = 3.1416f;

    armnn::TensorShape shape  ( 3, false );
    armnn::TensorInfo  info   ( shape, DataType::Float32 );
    armnn::Tensor      tensor ( info, &mutableDatum );

    BOOST_CHECK(armnn::Dimensionality::Specified == shape.GetDimensionality());
    BOOST_CHECK_MESSAGE(0 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    BOOST_CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    BOOST_CHECK(false == shape.GetDimensionSpecificity(0));
    BOOST_CHECK(false == shape.GetDimensionSpecificity(1));
    BOOST_CHECK(false == shape.GetDimensionSpecificity(2));
    BOOST_CHECK(!shape.AreAllDimensionsSpecified());
    BOOST_CHECK(!shape.IsAtLeastOneDimensionSpecified());

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    BOOST_TEST(shape_equal == shape);
    BOOST_TEST(shape_different != shape);
}

BOOST_AUTO_TEST_CASE(TensorShape_DynamicTensorType1_unknownSomeDimensionsSizes)
{
    std::vector<float> mutableDatum  { 42.f, 42.f, 42.f,
                                       0.0f, 0.1f, 0.2f };

    armnn::TensorShape shape         ( {2, 0, 3}, {true, false, true} );
    armnn::TensorInfo  info          ( shape, DataType::Float32 );
    armnn::Tensor      tensor        ( info, &mutableDatum );

    BOOST_CHECK(armnn::Dimensionality::Specified == shape.GetDimensionality());
    BOOST_CHECK_MESSAGE(6 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    BOOST_CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    BOOST_CHECK(true  == shape.GetDimensionSpecificity(0));
    BOOST_CHECK(false == shape.GetDimensionSpecificity(1));
    BOOST_CHECK(true  == shape.GetDimensionSpecificity(2));
    BOOST_CHECK(!shape.AreAllDimensionsSpecified());
    BOOST_CHECK(shape.IsAtLeastOneDimensionSpecified());

    BOOST_CHECK_THROW(shape[1], InvalidArgumentException);
    BOOST_CHECK_THROW(tensor.GetShape()[1], InvalidArgumentException);
    BOOST_CHECK_THROW(tensor.GetInfo().GetShape()[1], InvalidArgumentException);

    BOOST_TEST(2 == shape[0]);
    BOOST_TEST(2 == tensor.GetShape()[0]);
    BOOST_TEST(2 == tensor.GetInfo().GetShape()[0]);
    BOOST_CHECK_THROW( shape[1], InvalidArgumentException );

    BOOST_TEST(3 == shape[2]);
    BOOST_TEST(3 == tensor.GetShape()[2]);
    BOOST_TEST(3 == tensor.GetInfo().GetShape()[2]);

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    BOOST_TEST(shape_equal == shape);
    BOOST_TEST(shape_different != shape);
}

BOOST_AUTO_TEST_CASE(TensorShape_DynamicTensorType1_transitionFromUnknownToKnownDimensionsSizes)
{
    std::vector<float> mutableDatum  { 42.f, 42.f, 42.f,
                                       0.0f, 0.1f, 0.2f };

    armnn::TensorShape shape         (armnn::Dimensionality::NotSpecified );
    armnn::TensorInfo  info          ( shape, DataType::Float32 );
    armnn::Tensor      tensor        ( info, &mutableDatum );

    // Specify the number of dimensions
    shape.SetNumDimensions(3);
    BOOST_CHECK(armnn::Dimensionality::Specified == shape.GetDimensionality());
    BOOST_CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    BOOST_CHECK(false == shape.GetDimensionSpecificity(0));
    BOOST_CHECK(false == shape.GetDimensionSpecificity(1));
    BOOST_CHECK(false == shape.GetDimensionSpecificity(2));
    BOOST_CHECK(!shape.AreAllDimensionsSpecified());
    BOOST_CHECK(!shape.IsAtLeastOneDimensionSpecified());

    // Specify dimension 0 and 2.
    shape.SetDimensionSize(0, 2);
    shape.SetDimensionSize(2, 3);
    BOOST_CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    BOOST_CHECK_MESSAGE(6 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    BOOST_CHECK(true  == shape.GetDimensionSpecificity(0));
    BOOST_CHECK(false == shape.GetDimensionSpecificity(1));
    BOOST_CHECK(true  == shape.GetDimensionSpecificity(2));
    BOOST_CHECK(!shape.AreAllDimensionsSpecified());
    BOOST_CHECK(shape.IsAtLeastOneDimensionSpecified());

    info.SetShape(shape);
    armnn::Tensor tensor2( info, &mutableDatum );
    BOOST_TEST(2 == shape[0]);
    BOOST_TEST(2 == tensor2.GetShape()[0]);
    BOOST_TEST(2 == tensor2.GetInfo().GetShape()[0]);

    BOOST_CHECK_THROW(shape[1], InvalidArgumentException);
    BOOST_CHECK_THROW(tensor.GetShape()[1], InvalidArgumentException);
    BOOST_CHECK_THROW(tensor.GetInfo().GetShape()[1], InvalidArgumentException);

    BOOST_TEST(3 == shape[2]);
    BOOST_TEST(3 == tensor2.GetShape()[2]);
    BOOST_TEST(3 == tensor2.GetInfo().GetShape()[2]);

    armnn::TensorShape shape_equal;
    armnn::TensorShape shape_different;
    shape_equal = shape;
    BOOST_TEST(shape_equal == shape);
    BOOST_TEST(shape_different != shape);

    // Specify dimension 1.
    shape.SetDimensionSize(1, 5);
    BOOST_CHECK_MESSAGE(3 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    BOOST_CHECK_MESSAGE(30 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    BOOST_CHECK(true  == shape.GetDimensionSpecificity(0));
    BOOST_CHECK(true  == shape.GetDimensionSpecificity(1));
    BOOST_CHECK(true  == shape.GetDimensionSpecificity(2));
    BOOST_CHECK(shape.AreAllDimensionsSpecified());
    BOOST_CHECK(shape.IsAtLeastOneDimensionSpecified());
}

BOOST_AUTO_TEST_CASE(Tensor_emptyConstructors)
{
    auto shape = armnn::TensorShape();
    BOOST_CHECK_MESSAGE( 0 == shape.GetNumDimensions(), "Number of dimensions is " << shape.GetNumDimensions());
    BOOST_CHECK_MESSAGE( 0 == shape.GetNumElements(), "Number of elements is " << shape.GetNumElements());
    BOOST_CHECK( armnn::Dimensionality::Specified == shape.GetDimensionality());
    BOOST_CHECK( shape.AreAllDimensionsSpecified());
    BOOST_CHECK_THROW( shape[0], InvalidArgumentException );

    auto tensor = armnn::Tensor();
    BOOST_CHECK_MESSAGE( 0 == tensor.GetNumDimensions(), "Number of dimensions is " << tensor.GetNumDimensions());
    BOOST_CHECK_MESSAGE( 0 == tensor.GetNumElements(), "Number of elements is " << tensor.GetNumElements());
    BOOST_CHECK_MESSAGE( 0 == tensor.GetShape().GetNumDimensions(), "Number of dimensions is " <<
                        tensor.GetShape().GetNumDimensions());
    BOOST_CHECK_MESSAGE( 0 == tensor.GetShape().GetNumElements(), "Number of dimensions is " <<
                        tensor.GetShape().GetNumElements());
    BOOST_CHECK( armnn::Dimensionality::Specified == tensor.GetShape().GetDimensionality());
    BOOST_CHECK( tensor.GetShape().AreAllDimensionsSpecified());
    BOOST_CHECK_THROW( tensor.GetShape()[0], InvalidArgumentException );
}
BOOST_AUTO_TEST_SUITE_END()
