//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/Descriptors.hpp"
#include "armnn/Types.hpp"
%}

namespace std {
    %template() vector<unsigned int>;
    %template() vector<int>;
    %template() vector<pair<unsigned int, unsigned int>>;
    %template(TensorShapeVector) vector<armnn::TensorShape>;
}

%include "typemaps/vectors.i"
%include "stdint.i"

%typemap(out) const uint32_t*
%{
{
    auto len = arg1->GetNumViews();
    $result = PyList_New(len);
    if (!$result) {
        Py_XDECREF($result);
        return PyErr_NoMemory();
    }
    for (unsigned int i = 0; i < len; ++i) {

        PyList_SetItem($result, i, PyLong_FromUnsignedLong($1[i]));
    }
}
%}

namespace armnn
{

%list_to_vector( std::vector<unsigned int> );
%list_to_vector( std::vector<int> );
%list_to_vector( std::vector<std::pair<unsigned int, unsigned int>> );

%feature("docstring",
    "
    A configuration for the Activation layer. See `INetwork.AddActivationLayer()`.

    Contains:
        m_Function (int): Specifies the activation function to use.
                          (`ActivationFunction_Sigmoid`, `ActivationFunction_TanH`, `ActivationFunction_Linear`,
                          `ActivationFunction_ReLu`, `ActivationFunction_BoundedReLu`, `ActivationFunction_SoftReLu`,
                          `ActivationFunction_LeakyReLu`, `ActivationFunction_Abs`, `ActivationFunction_Sqrt`,
                          `ActivationFunction_Square`).
                          Default: `ActivationFunction_Sigmoid`.
        m_A (float): Alpha upper bound value used by the activation functions. (`ActivationFunction_BoundedReLu`,
                      `ActivationFunction_Linear`, `ActivationFunction_TanH`). Default: 0.
        m_B (float): Beta lower bound value used by the activation functions. (`ActivationFunction_BoundedReLu`,
                      `ActivationFunction_Linear`, `ActivationFunction_TanH`). Default: 0.

    ") ActivationDescriptor;
struct ActivationDescriptor
{
    ActivationDescriptor();

    ActivationFunction m_Function;
    float              m_A;
    float              m_B;

    bool operator ==(const ActivationDescriptor &rhs) const;
};


%feature("docstring",
    "
    A descriptor for the ArgMinMax layer. See `INetwork.AddArgMinMaxLayer()`.

    Contains:
        m_Function (int): Specify if the function is to find Min or Max with `ArgMinMaxFunction_Min` or `ArgMinMaxFunction_Max`.
                          Default: `ArgMinMaxFunction_Min`.
        m_Axis (int): Axis to reduce across the input tensor. Default: -1.

    ") ArgMinMaxDescriptor;
struct ArgMinMaxDescriptor
{
    ArgMinMaxDescriptor();

    ArgMinMaxFunction m_Function;
    int m_Axis;

    bool operator ==(const ArgMinMaxDescriptor &rhs) const;
};

%feature("docstring",
    "
    A descriptor for the BatchNormalization layer.  See `INetwork.AddBatchNormalizationLayer()`.

    Contains:
        m_Eps (float): Value to add to the variance. Used to avoid dividing by zero. Default: 0.0001.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.

    ") BatchNormalizationDescriptor;
struct BatchNormalizationDescriptor
{
    BatchNormalizationDescriptor();

    float m_Eps;
    DataLayout m_DataLayout;

    bool operator ==(const BatchNormalizationDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the BatchToSpaceNd layer.  See `INetwork.AddBatchToSpaceNdLayer()`.

    Contains:
        m_BlockShape (list of int): Block shape values. Default: (1, 1). Underlying C++ type is unsigned int.
        m_Crops (list of tuple): The values to crop from the input dimension. Default: [(0, 0), (0, 0)].
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.

    ") BatchToSpaceNdDescriptor;
struct BatchToSpaceNdDescriptor
{
    BatchToSpaceNdDescriptor();
    BatchToSpaceNdDescriptor(std::vector<unsigned int> blockShape,
                             std::vector<std::pair<unsigned int, unsigned int>> crops);

    std::vector<unsigned int> m_BlockShape;
    std::vector<std::pair<unsigned int, unsigned int>> m_Crops;
    DataLayout m_DataLayout;

    bool operator ==(const BatchToSpaceNdDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the ChannelShuffle layer.  See `INetwork.AddChannelShuffleLayer()`.

    Contains:
        m_NumGroups (int): Underlying C++ type is uint32_t. Number of groups for the shuffle operation. Default: 0.
        m_Axis (int): Underlying C++ type is uint32_t. 0-based axis along which shuffle is performed. Default: 0.

    ") ChannelShuffleDescriptor;
struct ChannelShuffleDescriptor
{
    ChannelShuffleDescriptor();
    ChannelShuffleDescriptor(int numGroups, int axis);

    int m_NumGroups;
    int m_Axis;

    bool operator ==(const ChannelShuffleDescriptor &rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Comparison layer.  See `INetwork.AddComparisonLayer()`.

    Contains:
        m_Operation (int): Specifies the comparison operation to execute.
                                           (`ComparisonOperation_Equal`, `ComparisonOperation_Greater`, `ComparisonOperation_GreaterOrEqual`,
                                           `ComparisonOperation_Less`, `ComparisonOperation_LessOrEqual`, `ComparisonOperation_NotEqual`)
                                           Default: `ComparisonOperation_Equal`.
    ") ComparisonDescriptor;
struct ComparisonDescriptor
{
    ComparisonDescriptor();

    ComparisonDescriptor(ComparisonOperation operation);

    bool operator ==(const ComparisonDescriptor &rhs) const;

    ComparisonOperation m_Operation;
};

%feature("docstring",
    "
    Creates a configuration/descriptor for a Concatenation layer. See `INetwork.AddConcatLayer()`.
    Number of Views must be equal to the number of inputs, and their order must match e.g. first view corresponds to the first input, second view to the second input, etc.

    Contains:
        numViews (int): Number of views, the value  must be equal to the number of outputs of a layer.
        numDimensions (int): Number of dimensions. Default value is 4.

    ") ConcatDescriptor;
struct ConcatDescriptor
{
    ConcatDescriptor();

    ConcatDescriptor(uint32_t numViews, uint32_t numDimensions = 4);

    %feature("docstring",
        "
        Get the number of views.
        Returns:
            int: Number of views.
        ") GetNumViews;
    uint32_t GetNumViews() const;

    %feature("docstring",
        "
        Get the number of dimensions.
        Returns:
            int: Number of dimensions.
        ") GetNumDimensions;
    uint32_t GetNumDimensions() const;

    %feature("docstring",
        "
        Get the view origin input by index.

        Each view match the inputs order, e.g. first view corresponds to the first input, second view to the second input, etc.

        Args:
            idx (int): Index to get view from.

        Returns:
            list: View origin (shape) specified by the int value `idx` as a list of ints.
        ") GetViewOrigin;

    const uint32_t* GetViewOrigin(uint32_t idx) const;

    %feature("docstring",
        "
        Set the concatenation dimension.
        Args:
            concatAxis (int): Concatenation axis index.
        ") SetConcatAxis;
    void SetConcatAxis(unsigned int concatAxis);

    %feature("docstring",
        "
        Get the concatenation dimension.
        Returns:
            int: Concatenation axis index.
        ") GetConcatAxis;
    unsigned int GetConcatAxis() const;

    bool operator ==(const ConcatDescriptor& rhs) const;
};
%extend ConcatDescriptor{
     %feature("docstring",
        "
        Set the coordinates of a specific origin view input.

        Args:
            view (int): Origin view index.
            coord (int): Coordinate of the origin view to set.
            value (int): Value to set.
        Raises:
            RuntimeError: If the `view` is greater than or equal to GetNumViews().
            RuntimeError: If the `coord` is greater than or equal to GetNumDimensions().
        ") SetViewOriginCoord;
    void SetViewOriginCoord(uint32_t view, uint32_t coord, uint32_t value) {
        armnn::Status status = $self->SetViewOriginCoord(view, coord, value);
        if(status == armnn::Status::Failure)
        {
            throw armnn::Exception("Failed to set view origin coordinates.");
        }
    };
}

%feature("docstring",
    "
    A descriptor for the Convolution2d layer.  See `INetwork.AddConvolution2dLayer()`.

    Contains:
        m_PadLeft (int): Underlying C++ data type is uint32_t. Padding left value in the width dimension. Default: 0.
        m_PadRight (int): Underlying C++ data type is uint32_t. Padding right value in the width dimension. Default: 0.
        m_PadTop (int): Underlying C++ data type is uint32_t. Padding top value in the height dimension. Default: 0.
        m_PadBottom (int): Underlying C++ data type is uint32_t. Padding bottom value in the height dimension. Default: 0.
        m_StrideX (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the width dimension. Default: 0.
        m_StrideY (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the height dimension. Default: 0.
        m_DilationX (int): Underlying C++ data type is uint32_t. Dilation along x axis. Default: 1.
        m_DilationY (int): Underlying C++ data type is uint32_t. Dilation along y axis. Default: 1.
        m_BiasEnabled (bool): Enable/disable bias. Default: false.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.

    ") Convolution2dDescriptor;
struct Convolution2dDescriptor
{
    Convolution2dDescriptor();

    uint32_t             m_PadLeft;
    uint32_t             m_PadRight;
    uint32_t             m_PadTop;
    uint32_t             m_PadBottom;
    uint32_t             m_StrideX;
    uint32_t             m_StrideY;
    uint32_t             m_DilationX;
    uint32_t             m_DilationY;
    bool                 m_BiasEnabled;
    DataLayout           m_DataLayout;

    bool operator ==(const Convolution2dDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Convolution3d layer.  See `INetwork.AddConvolution3dLayer()`.

    Contains:
        m_PadLeft (int): Underlying C++ data type is uint32_t. Padding left value in the width dimension. Default: 0.
        m_PadRight (int): Underlying C++ data type is uint32_t. Padding right value in the width dimension. Default: 0.
        m_PadTop (int): Underlying C++ data type is uint32_t. Padding top value in the height dimension. Default: 0.
        m_PadBottom (int): Underlying C++ data type is uint32_t. Padding bottom value in the height dimension. Default: 0.
        m_PadFront (int): Underlying C++ data type is uint32_t. Padding front value in the depth dimension. Default: 0.
        m_PadBack (int): Underlying C++ data type is uint32_t. Padding back value in the depth dimension. Default: 0.
        m_StrideX (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the width dimension. Default: 0.
        m_StrideY (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the height dimension. Default: 0.
        m_StrideZ (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the depth dimension. Default: 0.
        m_DilationX (int): Underlying C++ data type is uint32_t. Dilation along x axis. Default: 1.
        m_DilationY (int): Underlying C++ data type is uint32_t. Dilation along y axis. Default: 1.
        m_DilationZ (int): Underlying C++ data type is uint32_t. Dilation along z axis. Default: 1.
        m_BiasEnabled (bool): Enable/disable bias. Default: false.
        m_DataLayout (int): The data layout to be used (`DataLayout_NDHWC`, `DataLayout_NCDHW`). Default: 3 `DataLayout_NDHWC`.

    ") Convolution3dDescriptor;
struct Convolution3dDescriptor
{
    Convolution3dDescriptor();

    uint32_t             m_PadLeft;
    uint32_t             m_PadRight;
    uint32_t             m_PadTop;
    uint32_t             m_PadBottom;
    uint32_t             m_PadFront;
    uint32_t             m_PadBack;
    uint32_t             m_StrideX;
    uint32_t             m_StrideY;
    uint32_t             m_StrideZ;
    uint32_t             m_DilationX;
    uint32_t             m_DilationY;
    uint32_t             m_DilationZ;
    bool                 m_BiasEnabled;
    DataLayout           m_DataLayout;

    bool operator ==(const Convolution3dDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the DepthToSpace layer.  See `INetwork.AddDepthToSpaceLayer()`.

    Contains:
        m_BlockSize (int): Underlying C++ type is `unsigned int`.  Scalar specifying the input block size. It must be >= 1. Default: 1.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NHWC`.

    ") DepthToSpaceDescriptor;
struct DepthToSpaceDescriptor
{
    DepthToSpaceDescriptor();
    DepthToSpaceDescriptor(unsigned int blockSize, DataLayout dataLayout);

    unsigned int m_BlockSize;
    DataLayout m_DataLayout;
};


%feature("docstring",
    "
    A descriptor for the DepthwiseConvolution2d layer. See `INetwork.AddDepthwiseConvolution2dLayer()`.

    Contains:
        m_PadLeft (int): Underlying C++ data type is uint32_t. Padding left value in the width dimension. Default: 0.
        m_PadRight (int): Underlying C++ data type is uint32_t. Padding right value in the width dimension. Default: 0.
        m_PadTop (int): Underlying C++ data type is uint32_t. Padding top value in the height dimension. Default: 0.
        m_PadBottom (int): Underlying C++ data type is uint32_t. Padding bottom value in the height dimension. Default: 0.
        m_StrideX (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the width dimension. Default: 0.
        m_StrideY (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the height dimension. Default: 0.
        m_DilationX (int): Underlying C++ data type is uint32_t. Dilation along x axis. Default: 1.
        m_DilationY (int): Underlying C++ data type is uint32_t. Dilation along y axis. Default: 1.
        m_BiasEnabled (bool): Enable/disable bias. Default: false.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.

    ") DepthwiseConvolution2dDescriptor;
struct DepthwiseConvolution2dDescriptor
{
    DepthwiseConvolution2dDescriptor();

    uint32_t   m_PadLeft;
    uint32_t   m_PadRight;
    uint32_t   m_PadTop;
    uint32_t   m_PadBottom;
    uint32_t   m_StrideX;
    uint32_t   m_StrideY;
    uint32_t   m_DilationX;
    uint32_t   m_DilationY;
    bool       m_BiasEnabled;
    DataLayout m_DataLayout;

    bool operator ==(const DepthwiseConvolution2dDescriptor& rhs) const;

    uint32_t GetNumInputs() const;
};

%feature("docstring",
    "
    A descriptor for the DetectionPostProcess layer. See `INetwork.AddDetectionPostProcessLayer()`.

    This layer is a custom layer used to process the output from SSD MobilenetV1.

    Contains:
        m_MaxDetections (int): Underlying C++ data type is uint32_t. Maximum numbers of detections. Default: 0.
        m_MaxClassesPerDetection (int): Underlying C++ data type is uint32_t. Maximum numbers of classes per detection, used in Fast NMS. Default: 1.
        m_DetectionsPerClass (int): Underlying C++ data type is uint32_t. Detections per classes, used in Regular NMS. Default: 1.
        m_NmsScoreThreshold (float): Non maximum suppression score threshold. Default: 0.
        m_NmsIouThreshold (float): Intersection over union threshold. Default: 0.
        m_NumClasses (int): Underlying C++ data type is uint32_t. Number of classes. Default: 0.
        m_UseRegularNms (bool): Use Regular Non maximum suppression. Default: false.
        m_ScaleX (float): Center size encoding scale x. Default: 0.
        m_ScaleY (float): Center size encoding scale y. Default: 0.
        m_ScaleW (float): Center size encoding scale weight. Default: 0.
        m_ScaleH (float): Center size encoding scale height. Default: 0.

    ") DetectionPostProcessDescriptor;
struct DetectionPostProcessDescriptor
{
    DetectionPostProcessDescriptor();

    uint32_t m_MaxDetections;
    uint32_t m_MaxClassesPerDetection;
    uint32_t m_DetectionsPerClass;
    float m_NmsScoreThreshold;
    float m_NmsIouThreshold;
    uint32_t m_NumClasses;
    bool m_UseRegularNms;
    float m_ScaleX;
    float m_ScaleY;
    float m_ScaleW;
    float m_ScaleH;

    bool operator ==(const DetectionPostProcessDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the FakeQuantization layer.

    Contains:
        m_Min (float): Minimum value for quantization range. Default: -6.0.
        m_Max (float): Maximum value for quantization range. Default: 6.0.

    ") FakeQuantizationDescriptor;
struct FakeQuantizationDescriptor
{
    FakeQuantizationDescriptor();

    float m_Min;
    float m_Max;

    bool operator ==(const FakeQuantizationDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Fill layer. Creates a tensor filled with a scalar value.

    Contains:
        m_Value (float): Value the tensor will be filled with.

    ") FillDescriptor;
struct FillDescriptor
{
    FillDescriptor();
    FillDescriptor(const float& value);

    bool operator ==(const FillDescriptor& rhs) const;

    float m_Value;
};

%feature("docstring",
    "
    A descriptor for the Gather layer.

    Contains:
        m_Axis (int32_t): The axis from where to gather values from.

    ") GatherDescriptor;
struct GatherDescriptor
{
    GatherDescriptor();

    GatherDescriptor(int32_t axis);

    bool operator ==(const GatherDescriptor& rhs) const
    {
        return m_Axis == rhs.m_Axis;
    }

    int32_t m_Axis;
};

%feature("docstring",
    "
    A descriptor for the FullyConnected layer. See `INetwork.AddFullyConnectedLayer()`.

    Contains:
        m_BiasEnabled (bool): Enable/disable bias. Default: false.
        m_TransposeWeightMatrix (bool): Enable/disable transpose weight matrix. Default: false.

    ") FullyConnectedDescriptor;
struct FullyConnectedDescriptor
{
    FullyConnectedDescriptor();

    bool m_BiasEnabled;
    bool m_TransposeWeightMatrix;

    bool operator ==(const FullyConnectedDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for InstanceNormalization layer. See `INetwork.AddInstanceNormalizationLayer()`.

    Contains:
        m_Gamma (float): Gamma, the scale scalar value applied for the normalized tensor. Default: 1.0.
        m_Gamma (float): Beta, the offset scalar value applied for the normalized tensor. Default: 0.0.
        m_Gamma (float): Epsilon, small scalar value added to variance to avoid dividing by zero. Default: 1e-12.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.

    ") InstanceNormalizationDescriptor;
struct InstanceNormalizationDescriptor
{
    InstanceNormalizationDescriptor();

    float m_Gamma;
    float m_Beta;
    float m_Eps;
    DataLayout m_DataLayout;

    bool operator ==(const InstanceNormalizationDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the LSTM layer. See `INetwork.AddLstmLayer()`.

    Contains:
        m_ActivationFunc (int): Underlying C++ data type is uint32_t. The activation function to use. 0: None, 1: Relu, 3: Relu6, 4: Tanh, 6: Sigmoid.
                                     Default: 1.
        m_ClippingThresCell (float): Clipping threshold value for the cell state. Default: 0.0.
        m_ClippingThresProj (float): Clipping threshold value for the projection. Default: 0.0.
        m_CifgEnabled (bool): Enable/disable cifg (coupled input & forget gate). Default: true.
        m_PeepholeEnabled (bool): Enable/disable peephole. Default: false.
        m_ProjectionEnabled (bool): Enable/disable the projection layer. Default: false.
        m_LayerNormEnabled (bool): Enable/disable layer normalization. Default: false.
        m_TimeMajor (bool): Enable/disable time major. Default: false.
        m_InputIntermediateScale (float): Input intermediate quantization scale. Default: 0.0.
        m_ForgetIntermediateScale (float): Forget intermediate quantization scale. Default: 0.0.
        m_CellIntermediateScale (float): Cell intermediate quantization scale. Default: 0.0.
        m_OutputIntermediateScale (float): Output intermediate quantization scale. Default: 0.0.
        m_HiddenStateZeroPoint (int): Hidden State zero point. Default: 0.
        m_HiddenStateScale (float): Hidden State quantization scale. Default: 0.0.
    ") LstmDescriptor;
struct LstmDescriptor
{
    LstmDescriptor();

    uint32_t m_ActivationFunc;
    float m_ClippingThresCell;
    float m_ClippingThresProj;
    bool m_CifgEnabled;
    bool m_PeepholeEnabled;
    bool m_ProjectionEnabled;
    bool m_LayerNormEnabled;
    bool m_TimeMajor;
    float m_InputIntermediateScale;
    float m_ForgetIntermediateScale;
    float m_CellIntermediateScale;
    float m_OutputIntermediateScale;
    int32_t m_HiddenStateZeroPoint;
    float m_HiddenStateScale;

    bool operator ==(const LstmDescriptor& rhs) const;
};

%feature("docstring",
    "
    A Descriptor for the L2Normalization layer. See `INetwork.AddL2NormalizationLayer()`.

    Contains:
        m_Eps (float): Used to avoid dividing by zero. Default: 1e-12.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.

    ") L2NormalizationDescriptor;
struct L2NormalizationDescriptor
{
    L2NormalizationDescriptor();

    float m_Eps;
    DataLayout m_DataLayout;

    bool operator ==(const L2NormalizationDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Mean layer. See `INetwork.AddMeanLayer()`.

    Contains:
        m_Axis (list of int): Underlying C++ data type is std::vector<unsigned int>. Used to avoid dividing by zero. Values for the dimensions to reduce.
        m_KeepDims (bool): Enable/disable keep dimensions. If true, then the reduced dimensions that are of length 1 are kept. Default: False.

    ") MeanDescriptor;
struct MeanDescriptor
{
    MeanDescriptor();
    MeanDescriptor(const std::vector<unsigned int>& axis, bool keepDims);

    std::vector<unsigned int> m_Axis;
    bool m_KeepDims;

    bool operator ==(const MeanDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Normalization layer. See `INetwork.AddNormalizationLayer()`.

    Contains:
        m_NormChannelType (int): Normalization channel algorithm to use (`NormalizationAlgorithmMethod_Across`, `NormalizationAlgorithmMethod_Within`).
                                                           Default: `NormalizationAlgorithmChannel_Across`.
        m_NormMethodType (int): Normalization method algorithm to use (`NormalizationAlgorithmMethod_LocalBrightness`, `NormalizationAlgorithmMethod_LocalContrast`).
                                                         Default: `NormalizationAlgorithmMethod_LocalBrightness`.
        m_NormSize (int): Underlying C++ data type is uint32_t. Depth radius value. Default: 0.
        m_Alpha (float): Alpha value for the normalization equation. Default: 0.0.
        m_Beta (float): Beta value for the normalization equation. Default: 0.0.
        m_K (float): Kappa value used for the across channel normalization equation. Default: 0.0.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.

    ") NormalizationDescriptor;
struct NormalizationDescriptor
{
    NormalizationDescriptor();

    NormalizationAlgorithmChannel m_NormChannelType;
    NormalizationAlgorithmMethod  m_NormMethodType;
    uint32_t                      m_NormSize;
    float                         m_Alpha;
    float                         m_Beta;
    float                         m_K;
    DataLayout                    m_DataLayout;

    bool operator ==(const NormalizationDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Pad layer. See `INetwork.AddPadLayer()`.

    Contains:
        m_PadList (list of tuple): specifies the padding for input dimension.
                                   The first tuple value is the number of values to add before the tensor in the dimension.
                                   The second tuple value is the number of values to add after the tensor in the dimension.
                                   The number of pairs should match the number of dimensions in the input tensor.
        m_PadValue (bool): Optional value to use for padding. Default: 0.
        m_PaddingMode (int): The padding mode controls whether the padding should be filled
                             with constant values (`PaddingMode_Constant`), or reflect the input,
                             either excluding the border values (`PaddingMode_Reflect`)
                             or including them (`PaddingMode_Symmetric`).
                             Default: 0 (`PaddingMode_Constant`).

    ") PadDescriptor;
struct PadDescriptor
{
    PadDescriptor();
    PadDescriptor(const std::vector<std::pair<unsigned int, unsigned int>>& padList,
                  const float& padValue = 0,
                  const PaddingMode& paddingMode = PaddingMode_Constant);

    std::vector<std::pair<unsigned int, unsigned int>> m_PadList;
    float m_PadValue;
    PaddingMode m_PaddingMode;

    bool operator ==(const PadDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the ElementwiseUnary layer. See `INetwork.AddElementwiseUnaryLayer()`.

    Contains:
        m_Operation (int): Indicates which Unary operation to use. (`UnaryOperation_Abs`, `UnaryOperation_Exp`,
                           `UnaryOperation_Neg`, `UnaryOperation_Rsqrt`, `UnaryOperation_Sqrt`)
                           Default: `UnaryOperation_Abs`.

    ") ElementwiseUnaryDescriptor;
struct ElementwiseUnaryDescriptor
{
    ElementwiseUnaryDescriptor();
    ElementwiseUnaryDescriptor(UnaryOperation operation);

    UnaryOperation m_Operation;

    bool operator ==(const ElementwiseUnaryDescriptor &rhs) const;
};


%feature("docstring",
    "
    A descriptor for the Permute layer. See `INetwork.AddPermuteLayer()`.

    Contains:
        m_DimMappings (PermutationVector): Indicates how to translate tensor elements from a given source into the target destination,
                                           when source and target potentially have different memory layouts e.g. {0U, 3U, 1U, 2U}.

    ") PermuteDescriptor;
struct PermuteDescriptor
{
    PermuteDescriptor();
    PermuteDescriptor(const PermutationVector& dimMappings);

    PermutationVector m_DimMappings;

    bool operator ==(const PermuteDescriptor &rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Pooling2d layer. See `INetwork.AddPooling2dLayer()`.

    Contains:
        m_PoolType (int): The pooling algorithm to use (`PoolingAlgorithm_Max`, `PoolingAlgorithm_Average`, `PoolingAlgorithm_L2`). Default: `PoolingAlgorithm_Max`.
        m_PadLeft (int): Underlying C++ data type is uint32_t. Padding left value in the width dimension. Default: 0.
        m_PadRight (int): Underlying C++ data type is uint32_t. Padding right value in the width dimension. Default: 0.
        m_PadTop (int): Underlying C++ data type is uint32_t. Padding top value in the height dimension. Default: 0.
        m_PadBottom (int): Underlying C++ data type is uint32_t. Padding bottom value in the height dimension. Default: 0.
        m_PoolWidth (int): Underlying C++ data type is uint32_t. Pooling width value. Default: 0.
        m_PoolHeight (int): Underlying C++ data type is uint32_t. Pooling height value. Default: 0.
        m_StrideX (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the width dimension. Default: 0.
        m_StrideY (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the height dimension. Default: 0.
        m_OutputShapeRounding (int):  The rounding method for the output shape. (`OutputShapeRounding_Floor`, `OutputShapeRounding_Ceiling`).
                                                      Default: `OutputShapeRounding_Floor`.
        m_PaddingMethod (int): The padding method to be used. (`PaddingMethod_Exclude`, `PaddingMethod_IgnoreValue`).
                                         Default: `PaddingMethod_Exclude`.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.

    ") Pooling2dDescriptor;
struct Pooling2dDescriptor
{
    Pooling2dDescriptor();

    PoolingAlgorithm    m_PoolType;
    uint32_t            m_PadLeft;
    uint32_t            m_PadRight;
    uint32_t            m_PadTop;
    uint32_t            m_PadBottom;
    uint32_t            m_PoolWidth;
    uint32_t            m_PoolHeight;
    uint32_t            m_StrideX;
    uint32_t            m_StrideY;
    OutputShapeRounding m_OutputShapeRounding;
    PaddingMethod       m_PaddingMethod;
    DataLayout          m_DataLayout;

    bool operator ==(const Pooling2dDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Pooling3d layer. See `INetwork.AddPooling3dLayer()`.

    Contains:
        m_PoolType (int): The pooling algorithm to use (`PoolingAlgorithm_Max`, `PoolingAlgorithm_Average`, `PoolingAlgorithm_L2`). Default: `PoolingAlgorithm_Max`.
        m_PadLeft (int): Underlying C++ data type is uint32_t. Padding left value in the width dimension. Default: 0.
        m_PadRight (int): Underlying C++ data type is uint32_t. Padding right value in the width dimension. Default: 0.
        m_PadTop (int): Underlying C++ data type is uint32_t. Padding top value in the height dimension. Default: 0.
        m_PadBottom (int): Underlying C++ data type is uint32_t. Padding bottom value in the height dimension. Default: 0.
        m_PadFront (int): Underlying C++ data type is uint32_t. Padding front value in the depth dimension. Default: 0.
        m_PadBack (int): Underlying C++ data type is uint32_t. Padding back value in the depth dimension. Default: 0.
        m_PoolWidth (int): Underlying C++ data type is uint32_t. Pooling width value. Default: 0.
        m_PoolHeight (int): Underlying C++ data type is uint32_t. Pooling height value. Default: 0.
        m_StrideX (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the width dimension. Default: 0.
        m_StrideY (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the height dimension. Default: 0.
        m_StrideZ (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the depth dimension. Default: 0.
        m_OutputShapeRounding (int):  The rounding method for the output shape. (`OutputShapeRounding_Floor`, `OutputShapeRounding_Ceiling`).
                                                                                Default: `OutputShapeRounding_Floor`.
        m_PaddingMethod (int): The padding method to be used. (`PaddingMethod_Exclude`, `PaddingMethod_IgnoreValue`).
                                                                                Default: `PaddingMethod_Exclude`.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCDHW`, `DataLayout_NDHWC`). Default: `DataLayout_NCDHW`.

    ") Pooling3dDescriptor;
struct Pooling3dDescriptor
{
    Pooling3dDescriptor();

    PoolingAlgorithm    m_PoolType;
    uint32_t            m_PadLeft;
    uint32_t            m_PadRight;
    uint32_t            m_PadTop;
    uint32_t            m_PadBottom;
    uint32_t            m_PadFront;
    uint32_t            m_PadBack;
    uint32_t            m_PoolWidth;
    uint32_t            m_PoolHeight;
    uint32_t            m_StrideX;
    uint32_t            m_StrideY;
    uint32_t            m_StrideZ;
    OutputShapeRounding m_OutputShapeRounding;
    PaddingMethod       m_PaddingMethod;
    DataLayout          m_DataLayout;

    bool operator ==(const Pooling3dDescriptor& rhs) const;
};

%feature("docstring",
    "
    A ReduceDescriptor for the REDUCE operators.

    Contains:
        m_KeepDims(bool): If true then output shape has no change.
        m_vAxis (list of int): The indices of the dimensions to reduce.
        m_ReduceOperation (int): Specifies the reduction operation to execute ('ReduceOperation_Sum',
                                             'ReduceOperation_Max', 'ReduceOperation_Mean', 'ReduceOperation_Min',
                                             'ReduceOperation_Prod'). Default: 0 ('ReduceOperation_Sum').
        ") ReduceDescriptor;
struct ReduceDescriptor
{
    ReduceDescriptor();

    bool m_KeepDims;
    std::vector<uint32_t> m_vAxis;
    ReduceOperation m_ReduceOperation;

    bool operator ==(const ReduceDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Reshape layer. See `INetwork.AddReshapeLayer()`.

    Contains:
        m_TargetShape (TensorShape): Target shape value.

    ") ReshapeDescriptor;
struct ReshapeDescriptor
{
    ReshapeDescriptor();
    ReshapeDescriptor(const armnn::TensorShape& shape);

    armnn::TensorShape m_TargetShape;

    bool operator ==(const ReshapeDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Resize layer. See `INetwork.AddResizeLayer()`.

    Contains:
        m_TargetWidth (int): Underlying C++ data type is uint32_t. Target width value. Default: 0.
        m_TargetHeight (int): Underlying C++ data type is uint32_t. Target height value. Default: 0.
        m_Method (int): The Interpolation method to use (`ResizeMethod_Bilinear`, `ResizeMethod_NearestNeighbor`).
                        Default: `ResizeMethod_NearestNeighbor`.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.
        m_AlignCorners (bool):  Align corners or not when resizing. If True, corner pixel values are preserved after resizing.
                                        Default: False.
        m_HalfPixelCenters (bool): If true, calculates the pixels from the center instead of from the edge.
                                    Default: False.

    ") ResizeDescriptor;
struct ResizeDescriptor
{
    ResizeDescriptor();

    uint32_t m_TargetWidth;
    uint32_t m_TargetHeight;
    ResizeMethod m_Method;
    DataLayout m_DataLayout;
    bool m_AlignCorners;
    bool m_HalfPixelCenters;

    bool operator ==(const ResizeDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Slice layer. See `INetwork.AddSliceLayer()`.

    Contains:
        m_Begin (list of int): Underlying C++ data type is std::vector<unsigned int>. Beginning indices of the slice in each dimension.
        m_Size (list of int): Underlying C++ data type is std::vector<unsigned int>. Size of the slice in each dimension.

    ") SliceDescriptor;
struct SliceDescriptor
{
    SliceDescriptor();
    SliceDescriptor(const std::vector<unsigned int>& begin, const std::vector<unsigned int>& size);

    std::vector<unsigned int> m_Begin;
    std::vector<unsigned int> m_Size;

    bool operator ==(const SliceDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Space To Batch N-dimensions layer. See `INetwork.AddSpaceToBatchNdLayer()`.

    Contains:
        m_BlockShape (list of int): Underlying C++ data type is std::vector<unsigned int>. Block shape values. Default: [1, 1].
        m_Crops (list of tuple): Specifies the padding values for the input dimension:
                                 [heightPad - (top, bottom) widthPad - (left, right)].
                                 Default: [(0, 0), (0, 0)].
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.
    ") SpaceToBatchNdDescriptor;
struct SpaceToBatchNdDescriptor
{
    SpaceToBatchNdDescriptor();
    SpaceToBatchNdDescriptor(const std::vector<unsigned int>& blockShape,
                             const std::vector<std::pair<unsigned int, unsigned int>>& padList);

    std::vector<unsigned int> m_BlockShape;
    std::vector<std::pair<unsigned int, unsigned int>> m_PadList;
    DataLayout m_DataLayout;

    bool operator ==(const SpaceToBatchNdDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the SpaceToDepth layer. See `INetwork.AddSpaceToDepthLayer()`.

    Contains:
        m_BlockSize (int): Underlying C++ type is unsigned int.  Scalar specifying the input block size. It must be >= 1. Default: 1.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NHWC`.

    ") SpaceToDepthDescriptor;
struct SpaceToDepthDescriptor
{
    SpaceToDepthDescriptor();
    SpaceToDepthDescriptor(unsigned int blockSize, DataLayout dataLayout);

    unsigned int m_BlockSize;
    DataLayout m_DataLayout;

    bool operator ==(const SpaceToDepthDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for a Splitter layer. See `INetwork.AddSplitterLayer()`.

    Args:
        numViews (int): Number of views, the value  must be equal to the number of outputs of a layer.
        numDimensions (int): Number of dimensions. Default value is 4.

    ") SplitterDescriptor;
struct SplitterDescriptor
{

    SplitterDescriptor(uint32_t numViews, uint32_t numDimensions = 4);

    SplitterDescriptor();

    %feature("docstring",
        "
        Get the number of views.
        Returns:
            int: number of views.
        ") GetNumViews;
    uint32_t GetNumViews() const;

    %feature("docstring",
        "
        Get the number of dimensions.

        Returns:
            int: Number of dimensions.

        ") GetNumDimensions;
    uint32_t GetNumDimensions() const;

    %feature("docstring",
        "
        Get the output view origin (shape) by index, the order matches the outputs.

        e.g. first view corresponds to the first output, second view to the second output, etc.
        Args:
            idx (int): Index.
        Returns:
            list: View origin (shape) as a list of ints.
        ") GetViewOrigin;

    const uint32_t* GetViewOrigin(uint32_t idx) const;

    %feature("docstring",
        "
        Get the view sizes by index.
        Args:
            idx (int): Index.
        Returns:
            list: Sizes for the specified index as a list of ints.
        ") GetViewSizes;
    const uint32_t* GetViewSizes(uint32_t idx) const;


    %feature("docstring",
        "
        Get the view origins that describe how the splitting process is configured.

        The number of views is the number of outputs, and their order match.
        Returns:
            OriginsDescriptor: A descriptor for the origins view.
        ") GetOrigins;
    const ConcatDescriptor GetOrigins() const;

    bool operator ==(const SplitterDescriptor& rhs) const;
};

%extend SplitterDescriptor{
     %feature("docstring",
        "
        Set the value of a specific origin view input coordinate.

        Contains:
            view (int): Origin view index.
            coord (int): Coordinate of the origin view to set.
            value (int): Value to set.
        Raises:
            RuntimeError: If the `view` is greater than or equal to GetNumViews().
                          If the `coord` is greater than or equal to GetNumDimensions().
        ") SetViewOriginCoord;
    void SetViewOriginCoord(uint32_t view, uint32_t coord, uint32_t value) {
        armnn::Status status = $self->SetViewOriginCoord(view, coord, value);
        if(status == armnn::Status::Failure)
        {
            throw armnn::Exception("Failed to set view origin coordinates.");
        }
    };

    %feature("docstring",
        "
        Set the size of the views.

        Args:
            view (int): View index.
            coord (int): Coordinate of the origin view to set.
            value (int): Value to set.
        Raises:
            RuntimeError: If the `view` is greater than or equal to GetNumViews().
                          If the `coord` is greater than or equal to GetNumDimensions().
        ") SetViewSize;
    void SetViewSize(uint32_t view, uint32_t coord, uint32_t value) {
        armnn::Status status = $self->SetViewSize(view, coord, value);
        if(status == armnn::Status::Failure)
        {
            throw armnn::Exception("Failed to set view size.");
        }
    }
}

%feature("docstring",
    "
    A descriptor for the Stack layer. See `INetwork.AddStackLayer()`.

    Contains:
        m_Axis (int): Underlying C++ type is unsigned int. 0-based axis along which to stack the input tensors. Default: 0.
        m_NumInputs (int): Required shape of all input tensors. Default: 0.
        m_InputShape (TensorShape): Required shape of all input tensors.

    ") StackDescriptor;
struct StackDescriptor
{
    StackDescriptor();
    StackDescriptor(uint32_t axis, uint32_t numInputs, const armnn::TensorShape& inputShape);

    uint32_t m_Axis;
    uint32_t m_NumInputs;
    armnn::TensorShape m_InputShape;

    bool operator ==(const StackDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the StandIn layer. See `INetwork.AddStandInLayer()`.

    Contains:
        m_NumInputs (int): Underlying C++ type is unsigned int. Number of input tensors. Default: 0.
        m_NumOutputs (int): Underlying C++ type is unsigned int. Number of output tensors. Default: 0.

    ") StandInDescriptor;
struct StandInDescriptor
{
    StandInDescriptor();

    StandInDescriptor(uint32_t numInputs, uint32_t numOutputs);

    uint32_t m_NumInputs = 0;
    uint32_t m_NumOutputs = 0;
};

%feature("docstring",
    "
    A descriptor for the StridedSlice layer. See `INetwork.AddStridedSliceLayer()`.

    Contains:
        m_Begin (list of int): Underlying C++ data type is std::vector<int>. Begin values for the input that will be sliced.
        m_End (list of int): Underlying C++ data type is std::vector<int>. End values for the input that will be sliced.
        m_Stride (list of int): Underlying C++ data type is std::vector<int>. Stride values for the input that will be sliced.
        m_BeginMask (int): Underlying C++ data type is int32_t. Begin mask value. If set, then the begin is disregarded and
                               the fullest range is used for the dimension. Default: 0.
        m_EndMask (int): Underlying C++ data type is int32_t. End mask value. If set, then the end is disregarded and
                             the fullest range is used for the dimension.Default: 0.
        m_ShrinkAxisMask (int): Underlying C++ data type is int32_t. Shrink axis mask value. If set, the nth specification shrinks the dimensionality by 1. Default: 0.
        m_EllipsisMask (int): Underlying C++ data type is int32_t. Ellipsis mask value. Default: 0.
        m_NewAxisMask (int): Underlying C++ data type is int32_t. New axis mask value. If set, the begin, end and stride is disregarded and
                                  a new 1 dimension is inserted to this location of the output tensor. Default: 0.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.

    ") StridedSliceDescriptor;
struct StridedSliceDescriptor
{
    StridedSliceDescriptor();
    StridedSliceDescriptor(const std::vector<int> begin,
                           const std::vector<int> end,
                           const std::vector<int> stride);

    int GetStartForAxis(const armnn::TensorShape& inputShape, unsigned int axis) const;
    int GetStopForAxis(const armnn::TensorShape& inputShape, unsigned int axis, int startForAxis) const;

    std::vector<int> m_Begin;
    std::vector<int> m_End;
    std::vector<int> m_Stride;

    int32_t m_BeginMask;
    int32_t m_EndMask;
    int32_t m_ShrinkAxisMask;
    int32_t m_EllipsisMask;
    int32_t m_NewAxisMask;
    DataLayout m_DataLayout;

    bool operator ==(const StridedSliceDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Softmax layer. See `INetwork.AddSoftmaxLayer()`.

    Contains:
        m_Beta (float): Exponentiation value.
        m_Axis (int): Scalar, defaulted to the last index (-1), specifying the dimension the activation will be performed on.
    ") SoftmaxDescriptor;
struct SoftmaxDescriptor
{
    SoftmaxDescriptor();

    float m_Beta;
    int m_Axis;

    bool operator ==(const SoftmaxDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the TransposeConvolution2d layer. See `INetwork.AddTransposeConvolution2dLayer()`.

    Contains:
        m_PadLeft (int): Underlying C++ data type is uint32_t. Padding left value in the width dimension. Default: 0.
        m_PadRight (int): Underlying C++ data type is uint32_t. Padding right value in the width dimension. Default: 0.
        m_PadTop (int): Underlying C++ data type is uint32_t. Padding top value in the height dimension. Default: 0.
        m_PadBottom (int): Underlying C++ data type is uint32_t. Padding bottom value in the height dimension. Default: 0.
        m_StrideX (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the width dimension. Default: 0.
        m_StrideY (int): Underlying C++ data type is uint32_t. Stride value when proceeding through input for the height dimension. Default: 0.
        m_BiasEnabled (bool): Enable/disable bias. Default: false.
        m_DataLayout (int): The data layout to be used (`DataLayout_NCHW`, `DataLayout_NHWC`). Default: `DataLayout_NCHW`.
        m_OutputShapeEnabled (bool): Set to true if output shape is specified. Will prevent output shape inference.
        m_OutputShape (list of int): Output shape if it has been specified.

    ") TransposeConvolution2dDescriptor;
struct TransposeConvolution2dDescriptor
{
    TransposeConvolution2dDescriptor();

    uint32_t                  m_PadLeft;
    uint32_t                  m_PadRight;
    uint32_t                  m_PadTop;
    uint32_t                  m_PadBottom;
    uint32_t                  m_StrideX;
    uint32_t                  m_StrideY;
    bool                      m_BiasEnabled;
    DataLayout                m_DataLayout;
    bool                      m_OutputShapeEnabled;
    std::vector<unsigned int> m_OutputShape;

    bool operator ==(const TransposeConvolution2dDescriptor& rhs) const;
};

%feature("docstring",
    "
    A descriptor for the LogicalBinary layer. See `INetwork.AddLogicalBinaryLayer()`.

    Contains:
        m_Operation (int): Specifies the logical operation to execute.
                           (0: `LogicalBinaryOperation_LogicalAnd`, 1: `LogicalBinaryOperation_LogicalOr`)
                           Default: 0: `LogicalBinaryOperation_LogicalAnd`.

    ") LogicalBinaryDescriptor;
struct LogicalBinaryDescriptor
{
    LogicalBinaryDescriptor();
    LogicalBinaryDescriptor(LogicalBinaryOperation operation);

    LogicalBinaryOperation m_Operation;

    bool operator ==(const LogicalBinaryDescriptor &rhs) const;
};

%feature("docstring",
    "
    A descriptor for the Transpose layer. See `INetwork.AddTransposeLayer()`.

    Contains:
        m_DimMappings (PermutationVector): Indicates how to translate tensor elements from a given source into the target destination,
                                           when source and target potentially have different memory layouts e.g. {0U, 3U, 1U, 2U}.

    ") TransposeDescriptor;
struct TransposeDescriptor
{
    TransposeDescriptor();
    TransposeDescriptor(const PermutationVector& dimMappings);

    PermutationVector m_DimMappings;

    bool operator ==(const TransposeDescriptor &rhs) const;
};


using ConcatDescriptor = OriginsDescriptor;
using LogSoftmaxDescriptor = SoftmaxDescriptor;
using SplitterDescriptor = ViewsDescriptor;

%list_to_vector_clear(std::vector<unsigned int>);
%list_to_vector_clear(std::vector<int>);
%list_to_vector_clear(std::vector<std::pair<unsigned int, unsigned int>>);
}

%{
    armnn::ConcatDescriptor CreateDescriptorForConcatenation(std::vector<armnn::TensorShape> shapes,
                                       unsigned int concatenationDimension)
    {
        return  armnn::CreateDescriptorForConcatenation(shapes.begin(), shapes.end(), concatenationDimension);
    };
%}

%feature("docstring",
    "
    Create a descriptor for a Concatenation layer.
    Args:
        shapes (list of TensorShape): Input shapes of tensors to concatenated.
        concatenationDimension (int): Concatenation axis, must be >=0.

    Returns:
        ConcatDescriptor: A descriptor object for a Concatenation layer.
    ") CreateDescriptorForConcatenation;
armnn::ConcatDescriptor CreateDescriptorForConcatenation(std::vector<armnn::TensorShape> shapes,
                                                           unsigned int concatenationDimension);

%typemap(out) const uint32_t*;
