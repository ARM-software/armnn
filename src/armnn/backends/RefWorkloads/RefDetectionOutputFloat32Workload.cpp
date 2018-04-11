
#include "RefDetectionOutputFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{
    using uint = unsigned int;

    template <typename Dtype>
    class PredictionResult{
    public:
        int   classType;
        Dtype confidence;
        Dtype x;
        Dtype y;
        Dtype w;
        Dtype h;
        //Dtype objScore;
        //Dtype classScore;
        //Dtype confidence;
        //int classType;
    };

    template <typename Dtype>
    inline Dtype sigmoid(Dtype x)
    {
        return static_cast<Dtype>( 1. / (1. + exp(-x)) );
    }


    template <typename Dtype>
    inline void get_region_box(Dtype* x, PredictionResult<Dtype>& predict, std::vector<Dtype> biases,
                        uint n, uint index, uint i, uint j, uint w, uint h)
    {
        predict.x = ( static_cast<Dtype>(i) + sigmoid(x[index + 0]) ) / static_cast<Dtype>(w);
        predict.y = ( static_cast<Dtype>(j) + sigmoid(x[index + 1]) ) / static_cast<Dtype>(h);
        predict.w = static_cast<Dtype>( exp(x[index + 2]) * biases[2*n] / static_cast<Dtype>(w) );
        predict.h = static_cast<Dtype>( exp(x[index + 3]) * biases[2*n+1] / static_cast<Dtype>(h) );
    }

    template <typename Dtype>
    inline Dtype class_index_and_score(Dtype* input, uint classes, PredictionResult<Dtype>& predict)
    {
        Dtype sum = 0;
        Dtype large = input[0];
        uint classIndex = 0;
        for (uint i = 0; i < classes; ++i) {
            if (input[i] > large)
                large = input[i];
        }
        //large = *std::max_element<Dtype>(input ,input + classes);
        for (uint i = 0; i < classes; ++i) {
            Dtype e = static_cast<Dtype>( exp(input[i] - large) );
            sum += e;
            input[i] = e;
        }
        //std::accumulate<Dtype>(input ,input + classes ,large);
        //std::accumulate<Dtype>(input ,input + classes ,[](Dtype d){ return exp(d); });
        //sum = std::sum<Dtype>(input ,input + classes);
        //std::accumulate<Dtype>(input ,input + classes ,[sum](Dtype d){ return d/sum; });
        for (uint i = 0; i < classes; ++i) {
            input[i] = input[i] / sum;
        }
        large = input[0];
        classIndex = 0;

        for (uint i = 0; i < classes; ++i) {
            if (input[i] > large) {
                large = input[i];
                classIndex = i;
            }
        }
        //Dtype* pos = std::max_element<Dtype>(input ,input + classes);
        //predict.classScore = *pos;
        //predict.classType = std::distance<Dtype>(pos,input);
        predict.classType = static_cast<int>(classIndex);
        //predict.classScore = large;
        return large;
    }

    template<typename Dtype>
    inline void chop(Dtype *a)
    {
        Dtype xmin = std::max(Dtype(0),a[0] - a[2]/2.0f);
        Dtype xmax = std::min(Dtype(1),a[0] + a[2]/2.0f);
        a[0] = xmin;
        a[2] = xmax;
        Dtype ymin = std::max(Dtype(0),a[1] - a[3]/2.0f);
        Dtype ymax = std::min(Dtype(1),a[1] + a[3]/2.0f);
        a[1] = ymin;
        a[3] = ymax;
    }

    template<typename DType>
    inline DType CalculateOverlap( DType *a,  DType *b)
    {
        chop(a);
        chop(b);

        DType w = std::max(DType(0), std::min(a[2], b[2]) - std::max(a[0], b[0]));
        DType h = std::max(DType(0), std::min(a[3], b[3]) - std::max(a[1], b[1]));
        DType i = w * h;
        DType u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
        return u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
    }

    void applyNms(std::vector< PredictionResult<float> > &predicts,float nms_threshold)
    {
        uint valid_count = static_cast<uint>(predicts.size());
        float *p_out = reinterpret_cast<float*>(predicts.data());
        for (uint i = 0; i < valid_count; ++i)
        {
            uint offset_i = i * 6;
            if (p_out[offset_i] < 0)
                continue;  // skip eliminated
            for (uint j = i + 1; j < valid_count; ++j)
            {
                uint offset_j = j * 6;
                if (p_out[offset_j] < 0)
                    continue;  // skip eliminated
                //if (force_suppress || (p_out[offset_i] == p_out[offset_j]))
                if ( p_out[offset_i] == p_out[offset_j] )
                {
                    // when foce_suppress == true or class_id equals
                    float iou = CalculateOverlap(p_out + offset_i + 1, p_out + offset_j + 1);
                    if (iou >= nms_threshold)
                    {
                        p_out[offset_j] = -1;
                    }
                }
            }
        }
    }

    void RefDetectionOutputFloat32Workload::Execute() const
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDetectionOutputFloat32Workload_Execute");

        float*        outputData = GetOutputTensorDataFloat(0, m_Data);
        const float* inputData  = GetInputTensorDataFloat(0, m_Data);

        TensorInfo inInfo = GetTensorInfo(m_Data.m_Inputs[0]);
        TensorShape inShape = inInfo.GetShape();

        const unsigned int classes = m_Data.m_Parameters.m_Classes;
        const unsigned int side = m_Data.m_Parameters.m_Side;
        const unsigned int num_box = m_Data.m_Parameters.m_NumBox;

        std::vector<float> biases = m_Data.m_Parameters.m_Biases;
        const float confidence_threshold = m_Data.m_Parameters.m_ConfidenceThreshold;
        const float nms_threshold = m_Data.m_Parameters.m_NmsThreshold;


        const uint num = inShape[0];
        const uint height = inShape[2];
        const uint width = inShape[3];
        const uint channels = inShape[1];

        //CHECK_EQ((classes+5)*num_box,inShape[3]);

        //std::unique_ptr<float*> swap(malloc(num*height*width*channels*sizeof(float)));
        std::vector<float> swap(num*height*width*channels);

        auto data_at = [inputData,num,height,width,channels](uint b,uint c,uint h,uint w)->float
        {   return  inputData[b*height*width*channels + c*height*width + h*width + w];   };
        uint index = 0;
        for (uint b = 0 ; b < num ; ++b )
            for (uint h = 0 ; h < height; ++h )
                for (uint w = 0 ; w < width ; ++w )
                    for (uint c = 0 ; c < channels ; ++c )
                        swap[index++] = data_at(b, c, h, w);

        std::vector< PredictionResult<float> > predicts;
        PredictionResult<float> predict;
        predicts.clear();
        float objScore;
        float classScore;
        for (uint b = 0; b < num; ++b)
        {
            for (uint j = 0; j < side; ++j)
                for (uint i = 0; i < side; ++i)
                    for (uint n = 0; n < num_box; ++n)
                    {
                        uint index = b * channels * height * width +
                                    (j * side + i) * height * width + n * width;
                        //CHECK_EQ(swap[index], data_at(b, j * side + i, n, 0));

                        get_region_box<float>(swap.data(), predict, biases, n, index, i, j, side, side);

                        objScore = sigmoid(swap[index + 4]);
                        classScore = class_index_and_score(swap.data() + index + 5, classes, predict);

                        predict.confidence = objScore * classScore;
                        if (predict.confidence >= confidence_threshold)
                            predicts.push_back(predict);
                    }

            if (predicts.size() > 0)
                applyNms(predicts, nms_threshold);

            uint num_kept = 0;
            for (uint i = 0; i < predicts.size(); i++) {
                if (predicts[i].classType != -1) {
                    outputData[i * 7] = static_cast<float>(b);                         //Image_Id
                    outputData[i * 7 + 1] = static_cast<float>(predicts[i].classType); //label
                    outputData[i * 7 + 2] = predicts[i].confidence;                      //confidence
                    outputData[i * 7 + 3] = predicts[i].x;
                    outputData[i * 7 + 4] = predicts[i].y;
                    outputData[i * 7 + 5] = predicts[i].w;
                    outputData[i * 7 + 6] = predicts[i].h;
                    num_kept++;
                }
            }
            std::vector<uint> outShape(2, 1);
            outShape.push_back(num_kept);
            outShape.push_back(7);
            //SetTensorInfo(m_Data.m_Outputs[0]);
        }
    }

} //namespace armnn
