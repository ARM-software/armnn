//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/LstmParams.hpp"
#include "armnn/QuantizedLstmParams.hpp"
%}

namespace armnn
{

%feature("docstring",
    "
    Long Short-Term Memory layer input parameters.

    See `INetwork.AddLstmLayer()`.
    Operation described by the following equations:

     \[i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}C_{t-1}+b_i) \\\\
        f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}C_{t-1}+b_f) \\\\
        C_t=clip(f_t \odot C_{t-1} + i_t \odot g(W_{xc}x_t+W_{hc}h_{t-1}+b_c),\ t_{cell}) \\\\
        o_t = \sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o)  \\\\
        h_t = clip(W_{proj}(o_t \odot g(C_t))+b_{proj},\ t_{proj})\ if\ there\ is\ a\ projection;  \\\\
        h_t = o_t \odot g(C_t)\ otherwise. \]
        Where:
        \(x_t\) - input;
        \(i_t\) - input gate;
        \(f_t\) - forget gate;
        \(C_t\) - cell state;
        \(o_t\) - output;
        \(h_t\) - output state;
        \(\sigma\) - logistic sigmoid function;
        \(g\) - cell input and cell output activation function, see `LstmDescriptor.m_ActivationFunc`;
        \(t_{cell}\) - threshold for clipping the cell state, see `LstmDescriptor.m_ClippingThresCell`;
        \(t_{proj}\) - threshold for clipping the projected output, see `LstmDescriptor.m_ClippingThresProj`;

    Contains:
        m_InputToInputWeights (ConstTensor): \(W_{xi}\), input-to-input weight matrix.
        m_InputToForgetWeights (ConstTensor): \(W_{xf}\), input-to-forget weight matrix.
        m_InputToCellWeights (ConstTensor): \(W_{xc}\), input-to-cell weight matrix.
        m_InputToOutputWeights (ConstTensor): \(W_{xo}\), input-to-output weight matrix.

        m_RecurrentToInputWeights (ConstTensor): \(W_{hi}\), recurrent-to-input weight matrix.
        m_RecurrentToForgetWeights (ConstTensor): \(W_{hf}\), recurrent-to-forget weight matrix.
        m_RecurrentToCellWeights (ConstTensor): \(W_{hc}\), recurrent-to-cell weight matrix.
        m_RecurrentToOutputWeights (ConstTensor): \(W_{ho}\), recurrent-to-output weight matrix.

        m_CellToInputWeights (ConstTensor): \(W_{ci}\), cell-to-input weight matrix. Has effect if `LstmDescriptor.m_PeepholeEnabled`.
        m_CellToForgetWeights (ConstTensor): \(W_{cf}\), cell-to-forget weight matrix. Has effect if `LstmDescriptor.m_PeepholeEnabled`.
        m_CellToOutputWeights (ConstTensor): \(W_{co}\), cell-to-output weight matrix. Has effect if `LstmDescriptor.m_PeepholeEnabled`.

        m_InputGateBias (ConstTensor): \(b_i\), input gate bias.
        m_ForgetGateBias (ConstTensor): \(b_f\), forget gate bias.
        m_CellBias (ConstTensor): \(b_c\), cell bias.
        m_OutputGateBias (ConstTensor): \(b_o\),  output gate bias.

        m_ProjectionWeights (ConstTensor): \(W_{proj}\), projection weight matrix.
                                           Has effect if `LstmDescriptor.m_ProjectionEnabled` is set to True.
        m_ProjectionBias (ConstTensor): \(b_{proj}\), projection bias.
                                        Has effect if `LstmDescriptor.m_ProjectionEnabled` is set to True.
        m_InputLayerNormWeights (ConstTensor): normalisation weights for input,
                                               has effect if `LstmDescriptor.m_LayerNormEnabled` set to True.
        m_ForgetLayerNormWeights (ConstTensor): normalisation weights for forget gate,
                                                has effect if `LstmDescriptor.m_LayerNormEnabled` set to True.
        m_CellLayerNormWeights (ConstTensor): normalisation weights for current cell,
                                              has effect if `LstmDescriptor.m_LayerNormEnabled` set to True.
        m_OutputLayerNormWeights (ConstTensor): normalisation weights for output gate,
                                                has effect if `LstmDescriptor.m_LayerNormEnabled` set to True.

    ") LstmInputParams;
struct LstmInputParams
{
    LstmInputParams();

    const armnn::ConstTensor* m_InputToInputWeights;
    const armnn::ConstTensor* m_InputToForgetWeights;
    const armnn::ConstTensor* m_InputToCellWeights;
    const armnn::ConstTensor* m_InputToOutputWeights;
    const armnn::ConstTensor* m_RecurrentToInputWeights;
    const armnn::ConstTensor* m_RecurrentToForgetWeights;
    const armnn::ConstTensor* m_RecurrentToCellWeights;
    const armnn::ConstTensor* m_RecurrentToOutputWeights;
    const armnn::ConstTensor* m_CellToInputWeights;
    const armnn::ConstTensor* m_CellToForgetWeights;
    const armnn::ConstTensor* m_CellToOutputWeights;
    const armnn::ConstTensor* m_InputGateBias;
    const armnn::ConstTensor* m_ForgetGateBias;
    const armnn::ConstTensor* m_CellBias;
    const armnn::ConstTensor* m_OutputGateBias;
    const armnn::ConstTensor* m_ProjectionWeights;
    const armnn::ConstTensor* m_ProjectionBias;
    const armnn::ConstTensor* m_InputLayerNormWeights;
    const armnn::ConstTensor* m_ForgetLayerNormWeights;
    const armnn::ConstTensor* m_CellLayerNormWeights;
    const armnn::ConstTensor* m_OutputLayerNormWeights;
};

%feature("docstring",
    "
    Quantized Long Short-Term Memory layer input parameters.

    See `INetwork.AddQuantizedLstmLayer()`.
    Operation described by the following equations:

     \[i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}C_{t-1}+b_i) \\\\
        f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}C_{t-1}+b_f) \\\\
        C_t=clip(f_t \odot C_{t-1} + i_t \odot g(W_{xc}x_t+W_{hc}h_{t-1}+b_c),\ t_{cell}) \\\\
        o_t = \sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o)  \\\\
        h_t = clip(W_{proj}(o_t \odot g(C_t))+b_{proj},\ t_{proj})\ if\ there\ is\ a\ projection;  \\\\
        h_t = o_t \odot g(C_t)\ otherwise. \]
        Where:
        \(x_t\) - input;
        \(i_t\) - input gate;
        \(f_t\) - forget gate;
        \(C_t\) - cell state;
        \(o_t\) - output;
        \(h_t\) - output state;
        \(\sigma\) - logistic sigmoid function;
        \(g\) - cell input and cell output activation function, see `LstmDescriptor.m_ActivationFunc`;
        \(t_{cell}\) - threshold for clipping the cell state, see `LstmDescriptor.m_ClippingThresCell`;
        \(t_{proj}\) - threshold for clipping the projected output, see `LstmDescriptor.m_ClippingThresProj`;

    Contains:
        m_InputToInputWeights (ConstTensor): \(W_{xi}\), input-to-input weight matrix.
        m_InputToForgetWeights (ConstTensor): \(W_{xf}\), input-to-forget weight matrix.
        m_InputToCellWeights (ConstTensor): \(W_{xc}\), input-to-cell weight matrix.
        m_InputToOutputWeights (ConstTensor): \(W_{xo}\), input-to-output weight matrix.

        m_RecurrentToInputWeights (ConstTensor): \(W_{hi}\), recurrent-to-input weight matrix.
        m_RecurrentToForgetWeights (ConstTensor): \(W_{hf}\), recurrent-to-forget weight matrix.
        m_RecurrentToCellWeights (ConstTensor): \(W_{hc}\), recurrent-to-cell weight matrix.
        m_RecurrentToOutputWeights (ConstTensor): \(W_{ho}\), recurrent-to-output weight matrix.

        m_InputGateBias (ConstTensor): \(b_i\), input gate bias.
        m_ForgetGateBias (ConstTensor): \(b_f\), forget gate bias.
        m_CellBias (ConstTensor): \(b_c\), cell bias.
        m_OutputGateBias (ConstTensor): \(b_o\),  output gate bias.
    ") QuantizedLstmInputParams;
struct QuantizedLstmInputParams
{
    QuantizedLstmInputParams();

    const armnn::ConstTensor* m_InputToInputWeights;
    const armnn::ConstTensor* m_InputToForgetWeights;
    const armnn::ConstTensor* m_InputToCellWeights;
    const armnn::ConstTensor* m_InputToOutputWeights;
    const armnn::ConstTensor* m_RecurrentToInputWeights;
    const armnn::ConstTensor* m_RecurrentToForgetWeights;
    const armnn::ConstTensor* m_RecurrentToCellWeights;
    const armnn::ConstTensor* m_RecurrentToOutputWeights;
    const armnn::ConstTensor* m_InputGateBias;
    const armnn::ConstTensor* m_ForgetGateBias;
    const armnn::ConstTensor* m_CellBias;
    const armnn::ConstTensor* m_OutputGateBias;
};


}
