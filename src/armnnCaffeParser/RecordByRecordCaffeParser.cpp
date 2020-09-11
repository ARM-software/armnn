//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RecordByRecordCaffeParser.hpp"

#include "armnn/Exceptions.hpp"
#include "armnn/Utils.hpp"
#include <armnn/utility/NumericCast.hpp>

#include "GraphTopologicalSort.hpp"

// Caffe
#include <google/protobuf/wire_format.h>


//#include <stdio.h>
#include <limits.h>
#include <sstream>
//#include <iostream>
#include <fstream>

namespace armnnCaffeParser
{
// class which holds information on the absolute position in the stream
// of the data and the length of the data record.
class VarLenDataInfo
{
public:
    VarLenDataInfo(std::streamoff positionOfData, size_t sizeOfData) :
        m_PositionOfData(positionOfData), m_SizeOfData(sizeOfData) {}

    VarLenDataInfo(const VarLenDataInfo& x) :
        m_PositionOfData(x.PositionOfData()), m_SizeOfData (x.SizeOfData()) {}

    VarLenDataInfo& operator=(const VarLenDataInfo& x)
    {
        // handle self assignment
        if (this == &x) {
            return *this;
        }
        m_PositionOfData = x.PositionOfData(); m_SizeOfData = x.SizeOfData(); return *this;
    }

    std::streamoff PositionOfData() const {return m_PositionOfData;}
    size_t SizeOfData() const {return m_SizeOfData;}

private:
    std::streamoff m_PositionOfData;
    size_t m_SizeOfData;

};

// class which holds enough information on a LayerParameter in the Caffe protobuf
// format to allow it to be resolved for in place layering and sorted topologically
// prior to the entire record being parsed into memory.
//
// NOTE: function naming follows that of the protobuf classes these proxies are standing in for
class LayerParameterInfo : public VarLenDataInfo
{
public:
    static const std::string INPUT;
    LayerParameterInfo(const VarLenDataInfo& varLenDataInfo) :
        VarLenDataInfo(varLenDataInfo.PositionOfData(), varLenDataInfo.SizeOfData()),
        m_newTops(false), m_newBottoms(false) {}

    LayerParameterInfo(std::streamoff positionOfData, size_t sizeOfData) :
        VarLenDataInfo(positionOfData, sizeOfData), m_newTops(false), m_newBottoms(false) {}

    LayerParameterInfo(const LayerParameterInfo& x) :
        VarLenDataInfo(x.PositionOfData(), x.SizeOfData()),
        m_name(x.m_name),
        m_type(x.m_type),
        m_tops(x.m_tops),
        m_bottoms(x.m_bottoms),
        m_newTops(x.m_newTops),
        m_newBottoms(x.m_newBottoms) {}

    LayerParameterInfo& operator=(const LayerParameterInfo& x)
    {
        if (this == &x) {
            return *this;
        }
        VarLenDataInfo::operator=(x);
        m_name = x.m_name;
        m_type = x.m_type;
        m_tops = x.m_tops;
        m_bottoms = x.m_bottoms;
        m_newTops = x.m_newTops;
        m_newBottoms = x.m_newBottoms;
        return *this;
    }

    const std::string name() const {return m_name;}
    void set_name(const std::unique_ptr<char[]>& theName, size_t length)
    {
        m_name = std::string(theName.get(), length);
    }
    void set_name(const std::string& theName) {m_name = theName;}

    const std::string type() const {return m_type;}
    void set_type(const std::unique_ptr<char[]>& theType, size_t length)
    {
        m_type = std::string(theType.get(), length);
    }
    void set_type(const std::string& theType) {m_type = theType;}

    void add_top(const std::unique_ptr<char[]>& top, size_t length)
    {
        std::string topName(top.get(), length);
        m_tops.push_back(topName);
    }
    void add_top(const std::string& topName)
    {
        m_tops.push_back(topName);
    }
    const std::string top(unsigned long i) const {return m_tops[i];}
    unsigned long top_size() const {return m_tops.size();}
    void set_top(unsigned long i, const std::string& newName) {m_tops[i] = newName; m_newTops = true;}
    bool new_tops() const {return m_newTops;}

    void add_bottom(const std::unique_ptr<char[]>& bottom, size_t length)
    {
        std::string bottomName(bottom.get(), length);
        m_bottoms.push_back(bottomName);
    }
    unsigned long bottom_size() const {return m_bottoms.size();}
    const std::string bottom(unsigned long i) const {return m_bottoms[i];}
    void set_bottom(unsigned long i, const std::string& newName) {m_bottoms[i] = newName; m_newBottoms = true;}
    bool new_bottoms() const {return m_newBottoms;}

    // if the position and size of the data is zero and the type is "Input" then this is an 'Implicit Input Layer'
    // and needs to be handled differently from ordinary layers.
    bool isImplicitInputLayer() const
    {
        if ((PositionOfData() == 0) && (SizeOfData() == 0) && INPUT.compare(type()) == 0)
        {return true;} else {return false;}
    }

private:
    std::string m_name;
    std::string m_type;
    std::vector<std::string> m_tops;
    std::vector<std::string> m_bottoms;
    // mark the layers whose topology was changed
    // by the ResolveInPlaceLayers method.
    bool m_newTops;
    bool m_newBottoms;
};

// class which holds the field type (wire type) and field id (id from the .proto schema)
// read from the protobuf messages as per the binary encoding described in
// https://developers.google.com/protocol-buffers/docs/encoding
//
// NOTE: function naming follows that of the protobuf classes these proxies are standing in for
class ProtobufFieldInfo
{
public:
    ProtobufFieldInfo(int field_type, int field_id) :
        m_eof(false), m_field_type(field_type), m_field_id(field_id) {}
    ProtobufFieldInfo() : m_eof(true), m_field_type(0), m_field_id(0) {}

    bool eof() {return m_eof;}
    int field_type() {return m_field_type;}
    int field_id() {return m_field_id;}

private:
    bool m_eof;
    int m_field_type;
    int m_field_id;
};


// There are some NetParameter level data which are required
// to correctly processes some Caffe models. Specifically those which
// have 'implicit' input layers. Also it is nice to have the name of the model.
//
// NOTE: function naming follows that of the protobuf classes these proxies are standing in for
class NetParameterInfo
{
public:
    const std::string name() const {return m_name;}
    void set_name(const std::unique_ptr<char[]>&  theName, size_t length)
    {
        m_name = std::string(theName.get(), length);
    }

    void add_input(const std::unique_ptr<char[]>&  input, size_t length)
    {
        std::string inputName(input.get(), length);
        m_inputs.push_back(inputName);
    }
    const std::string input(unsigned long i) const {return m_inputs[i];}
    unsigned long input_size() const {return m_inputs.size();}

    void add_input_dimension(int input_dimension) {
        m_input_dimensions.push_back(input_dimension);
    }
    int input_dimension(unsigned long i) const {return m_input_dimensions[i];}
    unsigned long input_dimensions_size() const {return m_input_dimensions.size();}

    void add_blob_shape(caffe::BlobShape shape) {
        m_blob_shapes.push_back(shape);
    }
    const caffe::BlobShape blob_shape(unsigned long i) const {return m_blob_shapes[i];}
    unsigned long blob_shapes_size() const {return m_blob_shapes.size();}

private:
    std::string m_name;
    std::vector<std::string> m_inputs;
    std::vector<int> m_input_dimensions;
    std::vector<caffe::BlobShape> m_blob_shapes;

};

}; // namespace armnnCaffeParser

using namespace armnnCaffeParser;

// Initialise the class const
const std::string LayerParameterInfo::INPUT = "Input";

namespace
{

ProtobufFieldInfo readFieldInfo(std::ifstream& ifs)
{
    unsigned char first_byte = static_cast<unsigned char>(ifs.get());
    if (!ifs.good())
    {
        ProtobufFieldInfo eof;
        return eof;
    }
    int field_type = first_byte&7;
    int field_id = first_byte>>3;
    if ((field_id & 16) == 16)
    {
        unsigned char second_byte = static_cast<unsigned char>(ifs.get());
        if (!ifs.good())
        {
            ProtobufFieldInfo eof;
            return eof;
        }
        field_id = (field_id-16) + ((second_byte&127)<<4);
    }
    ProtobufFieldInfo fieldInfo(field_type, field_id);
    return fieldInfo;
}

const static int MAX_NUM_BYTES = 5;

int ReadBase128(std::ifstream& ifs)
{
    int result = 0;
    unsigned int shift_by = 0;
    int bytesRead = 0;
    while (true)
    {
        unsigned char a_byte = static_cast<unsigned char>(ifs.get());
        ++bytesRead;
        if (bytesRead > MAX_NUM_BYTES)
        {
            throw armnn::ParseException(
                "ReadBase128 exceeded the maximum number of bytes expected for an integer representation");
        }
        result += (a_byte & 127) << shift_by;
        shift_by += 7;
        if ((a_byte & 128) != 128)
        {
            break;
        }
    }
    return result;
}


std::unique_ptr<char[]> AllocateBuffer(std::ifstream& ifs, VarLenDataInfo& dataInfo)
{
    std::unique_ptr<char[]> ptr(new char[dataInfo.SizeOfData()]);
    ifs.clear();
    ifs.seekg(dataInfo.PositionOfData(), std::ios_base::beg);
    ifs.read(ptr.get(), armnn::numeric_cast<std::streamsize>(dataInfo.SizeOfData()));
    return ptr;
}

VarLenDataInfo CreateVarLenDataInfo(std::streamoff bufferStart, std::streamoff endOfLayer) {
    std::streamoff sizeOfLayer = endOfLayer - bufferStart;
    if (sizeOfLayer < 0)
    {
        std::stringstream ss;
        ss << "error when determining buffer size, negative value [" << sizeOfLayer << "]";
        throw armnn::ParseException(ss.str());
    }
    // NOTE: as some of the data being read in will be translated into strings (names of layers etc)
    //       the maximum size we can deal with is the upper size limit of a string i.e. size_t
    // on the platform in which I am currently compiling std::streamoff is signed long int and
    // size_t is unsigned long int so there is no way this error condition can fire but this stuff
    // is supposed to be portable so the check remains in place
    if (armnn::numeric_cast<size_t>(sizeOfLayer) > SIZE_MAX) {
        std::stringstream ss;
        ss << "layer is greater than " << SIZE_MAX << " in size cannot process. layer size = [" << sizeOfLayer << "]";
        throw armnn::ParseException(ss.str());
    }
    LayerParameterInfo info(bufferStart, armnn::numeric_cast<size_t>(sizeOfLayer));
    return info;
}

void ReadTopologicalInfoForLayerParameter(LayerParameterInfo& layerInfo, std::ifstream& ifs)
{
    // position the file pointer to the start of the layer data
    ifs.clear();
    ifs.seekg(layerInfo.PositionOfData(), std::ios_base::beg);
    std::streamoff endOfLayer = layerInfo.PositionOfData() +
        armnn::numeric_cast<std::streamoff>(layerInfo.SizeOfData());
    while(true)
    {
        // check to see if we have reached the end of the record
        std::streamoff currentPosition = ifs.tellg();
        if (currentPosition >= endOfLayer) {
            return;
        }
        // read the information for the next field.
        ProtobufFieldInfo fieldInfo = readFieldInfo(ifs);
        if (fieldInfo.eof())
        {
            return;
            // TODO: figure out whether this is an error condition or not...
            //throw armnn::ParseException("failed to read field from LayerParameter data");
        }
        // process the field
        switch (fieldInfo.field_type())
        {
            case 0:
            {
                ReadBase128(ifs);
                break;
            }
            case 2:
            {
                int size = ReadBase128(ifs);
                std::streamoff posStartOfData = ifs.tellg();
                VarLenDataInfo dataInfo(posStartOfData, armnn::numeric_cast<size_t>(size));
                //optional string name = 1; // the layer name
                //optional string type = 2; // the layer type
                //repeated string bottom = 3; // the name of each bottom blob
                //repeated string top = 4; // the name of each top blob
                if (fieldInfo.field_id() == 1)
                {
                    // read and set the name of the layer
                    auto layerName = AllocateBuffer(ifs, dataInfo);
                    layerInfo.set_name(layerName, dataInfo.SizeOfData());
                }
                else if (fieldInfo.field_id() == 2)
                {
                    // read and set the type of the layer
                    auto layerType = AllocateBuffer(ifs, dataInfo);
                    layerInfo.set_type(layerType, dataInfo.SizeOfData());
                }
                else if (fieldInfo.field_id() == 3)
                {
                    // read and add a bottom to the layer
                    auto bottom = AllocateBuffer(ifs, dataInfo);
                    layerInfo.add_bottom(bottom, dataInfo.SizeOfData());
                }
                else if (fieldInfo.field_id() == 4)
                {
                    // read and add a top to the layer
                    auto top = AllocateBuffer(ifs, dataInfo);
                    layerInfo.add_top(top, dataInfo.SizeOfData());
                }
                else
                {
                    ifs.seekg(size, std::ios_base::cur);
                    if (!ifs.good())
                    {
                        // TODO: error out?
                        return;
                    }
                }
                break;
            }
            case 1:
            {
                // 64 bit
                // advance by eight bytes
                ifs.seekg(8, std::ios_base::cur);
                if (!ifs.good())
                {
                    // TODO: error out?
                    return;
                }
                break;
            }
            case 5:
            {
                // 32 bit
                // advance by four bytes
                ifs.seekg(4, std::ios_base::cur);
                if (!ifs.good())
                {
                    // TODO: error out?
                    return;
                }
                break;
            }
            default:
            {
                throw armnn::ParseException("Encounted an unknown field type");
                break;
            }
        }
    }
}

void ResolveInPlaceLayers(std::vector<LayerParameterInfo>& layerInfo)
{
    std::map<std::string, std::vector<LayerParameterInfo*>> layersByTop;
    for (auto& info : layerInfo)
    {
        for (unsigned long i = 0; i < info.top_size(); ++i)
        {
            layersByTop[info.top(i)].push_back(&info);
        }
    }
    // For each set of layers with the same top, resolve them to a linear chain rather than in-place layers.
    // Note that for 'regular' layers, there will be a single layer in each group and so this will be a no-op.
    for (auto& layersWithSameTopIterator : layersByTop)
    {
        const std::string& top = layersWithSameTopIterator.first;
        const std::vector<LayerParameterInfo*> layersWithSameTop = layersWithSameTopIterator.second;

        // Chain the layers together in the order that they are listed in the prototxt (hopefully this is correct).
        // Note that the last layer will not have its top modified so that other layers will continue to reference it.
        for (unsigned int layerIdx = 0; layerIdx < layersWithSameTop.size() - 1; ++layerIdx)
        {
            LayerParameterInfo* layer1 = layersWithSameTop[layerIdx];
            LayerParameterInfo* layer2 = layersWithSameTop[layerIdx + 1];
            if (layer1->top_size() != 1)
            {
                throw armnn::ParseException("Node '" + layer1->name() + "' is an in-place layer but "
                                                                        "doesn't have exactly one top.");
            }
            std::string newTop = layer1->name() + "_top";
            layer1->set_top(0, newTop);
            if (layer2->bottom_size() != 1 || layer2->bottom(0) != top)
            {
                throw armnn::ParseException("Node '" + layer2->name() + "' is an in-place layer but "
                    " doesn't have exactly one bottom, or it doesn't match its top.");
            }
            layer2->set_bottom(0, newTop);

        }
    }
}

} // anonymous namespace, can't be seen outside this source file

RecordByRecordCaffeParser::RecordByRecordCaffeParser() : CaffeParserBase()
{}

armnn::INetworkPtr RecordByRecordCaffeParser::CreateNetworkFromBinaryFile(
    const char* graphFile,
    const std::map<std::string, armnn::TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{

    m_InputShapes = inputShapes;
    if (requestedOutputs.size() == 0)
    {
        throw armnn::ParseException("requestedOutputs must have at least one entry");
    }
    m_RequestedOutputs = requestedOutputs;

    std::ifstream ifs(graphFile, std::ifstream::in|std::ifstream::binary);
    if (ifs.fail())
    {
        throw armnn::FileNotFoundException("Failed to open graph file '" + std::string(graphFile) + "'");
    }

    std::vector<LayerParameterInfo> layerInfo;
    NetParameterInfo netParameterInfo;
    while(true)
    {
        ProtobufFieldInfo fieldInfo = readFieldInfo(ifs);
        if (fieldInfo.eof())
        {
            break;
        }
        switch(fieldInfo.field_type())
        {
            case 0:
            {
                ReadBase128(ifs);
                break;
            }
            case 2:
            {
                // The values of interest from the caffe.proto schema are:
                // optional string name = 1; // consider giving the network a name
                // DEPRECATED. See InputParameter. The input blobs to the network.
                // repeated string input = 3;
                // DEPRECATED. See InputParameter. The shape of the input blobs.
                // repeated BlobShape input_shape = 8;

                // 4D input dimensions -- deprecated.  Use "input_shape" instead.
                // If specified, for each input blob there should be four
                // values specifying the num, channels, height and width of the input blob.
                // Thus, there should be a total of (4 * #input) numbers.
                // repeated int32 input_dim = 4;

                // The layers that make up the net.  Each of their configurations, including
                // connectivity and behavior, is specified as a LayerParameter.
                // repeated LayerParameter layer = 100;  // ID 100 so layers are printed last.

                // The first four will (if present) be read into the NetParameterInfo
                // the LayerParameters will be read into the LayerParameterInfo vector.

                int size = ReadBase128(ifs);
                std::streamoff posStartOfData = ifs.tellg();
                ifs.seekg(size, std::ios_base::cur);
                if(!ifs.good())
                {
                    throw armnn::ParseException("failed to seek ahead in binary caffe file");
                }
                std::streamoff endOfLayer = ifs.tellg();
                if (fieldInfo.field_id() == 1)
                {
                    VarLenDataInfo dataInfo = CreateVarLenDataInfo(posStartOfData, endOfLayer);
                    auto graphName = AllocateBuffer(ifs, dataInfo);
                    netParameterInfo.set_name(graphName, dataInfo.SizeOfData());
                }
                if (fieldInfo.field_id() == 3)
                {
                    VarLenDataInfo dataInfo = CreateVarLenDataInfo(posStartOfData, endOfLayer);
                    auto inputName = AllocateBuffer(ifs, dataInfo);
                    netParameterInfo.add_input(inputName, dataInfo.SizeOfData());
                }
                if (fieldInfo.field_id() == 8)
                {
                    VarLenDataInfo dataInfo = CreateVarLenDataInfo(posStartOfData, endOfLayer);
                    auto inputShape = AllocateBuffer(ifs, dataInfo);
                    caffe::BlobShape blobShape;
                    bool bRet = blobShape.ParseFromArray(inputShape.get(), static_cast<int>(dataInfo.SizeOfData()));
                    if (!bRet)
                    {
                        throw armnn::ParseException("Failed to parse input shape");
                    }
                    netParameterInfo.add_blob_shape(blobShape);
                }
                if (fieldInfo.field_id() == 4)
                {
                    int input_dim = ReadBase128(ifs);
                    netParameterInfo.add_input_dimension(input_dim);
                }
                if (fieldInfo.field_id() == 100)
                {
                    LayerParameterInfo info(CreateVarLenDataInfo(posStartOfData, endOfLayer));
                    ReadTopologicalInfoForLayerParameter(info, ifs);
                    layerInfo.push_back(info);
                }
                break;
            }
            default:
            {
                break;
            }
        }
    }
    std::vector<const LayerParameterInfo*> sortedNodes;
    ProcessLayers(netParameterInfo, layerInfo, m_RequestedOutputs, sortedNodes);
    armnn::INetworkPtr networkPtr = LoadLayers(ifs, sortedNodes, netParameterInfo);
    return networkPtr;

}

void RecordByRecordCaffeParser::ProcessLayers(
    const NetParameterInfo& netParameterInfo,
    std::vector<LayerParameterInfo>& layerInfo,
    const std::vector<std::string>& m_RequestedOutputs,
    std::vector<const LayerParameterInfo*>& sortedNodes)
{
    // if there is an implicit input layer add it to the layerInfo list
    if (netParameterInfo.input_size() > 0)
    {
        LayerParameterInfo implicitInputLayer(0, 0);
        implicitInputLayer.set_type(LayerParameterInfo::INPUT);
        implicitInputLayer.set_name(netParameterInfo.input(0));
        implicitInputLayer.add_top(netParameterInfo.input(0));
        layerInfo.push_back(implicitInputLayer);
    }
    ::ResolveInPlaceLayers(layerInfo);

    for (LayerParameterInfo& info : layerInfo)
    {
        for (unsigned long i = 0; i < info.top_size(); ++i)
        {
            m_CaffeLayersByTopName[info.top(i)] = &info;
        }
    }

    // Find the output layers the user requested
    std::vector<const LayerParameterInfo*> targetLayers;
    for (const std::string& requestedOutputName : m_RequestedOutputs)
    {
        auto nodeIt = m_CaffeLayersByTopName.find(requestedOutputName);
        if (nodeIt == m_CaffeLayersByTopName.end())
        {
            throw armnn::ParseException(
                "Couldn't find requested output layer '" + requestedOutputName + "' in graph");
        }
        targetLayers.push_back(nodeIt->second);
    }

    // Sort them into a linear ordering such that all inputs of a node are before the node itself
    if (!armnnUtils::GraphTopologicalSort<const LayerParameterInfo*>(
        targetLayers,
        [this](const LayerParameterInfo* node)
            {
                return GetInputs(*node);
            },
        sortedNodes))
    {
        throw armnn::ParseException("Cycle detected in graph");
    }
}


std::vector<const LayerParameterInfo*> RecordByRecordCaffeParser::GetInputs(
    const LayerParameterInfo& layerParam)
{
    std::vector<const LayerParameterInfo*> ret;
    ret.reserve(layerParam.bottom_size());
    for (unsigned long j = 0; j < layerParam.bottom_size(); ++j)
    {
        std::string inputName = layerParam.bottom(j);
        auto inputIt = m_CaffeLayersByTopName.find(inputName);
        if (inputIt == m_CaffeLayersByTopName.end())
        {
            throw armnn::ParseException(
                "Can't find Caffe layer with top called '" + inputName + "', which is listed as an input of '" +
                layerParam.name() + "'");
        }
        ret.push_back(inputIt->second);
    }

    return ret;
}

armnn::INetworkPtr RecordByRecordCaffeParser::LoadLayers(std::ifstream& ifs,
                                                         std::vector<const LayerParameterInfo *>& sortedNodes,
                                                         const NetParameterInfo& netParameterInfo)
{

    m_NetworkInputsBindingInfo.clear();
    m_NetworkOutputsBindingInfo.clear();

    m_Network = armnn::INetwork::Create();

    for (auto info : sortedNodes)
    {
        caffe::LayerParameter layer;
        if (info->isImplicitInputLayer())
        {
            // create the matching Layer Parameter programatically from the data in the
            // net parameter info which has been passed in...
            layer.set_type(LayerParameterInfo::INPUT);
            layer.set_name(netParameterInfo.input(0));
            layer.add_top(netParameterInfo.input(0));

            caffe::InputParameter* inputParam = layer.mutable_input_param();
            caffe::BlobShape* shape = inputParam->add_shape();

            long unsigned int dim_size = netParameterInfo.input_dimensions_size();
            for (long unsigned int i = 0; i < dim_size; ++i)
            {
                shape->add_dim(netParameterInfo.input_dimension(i));
            }
        }
        else
        {
            char *buffer = new char[info->SizeOfData()];
            ifs.clear();
            ifs.seekg(info->PositionOfData(), std::ios_base::beg);
            ifs.read(buffer, armnn::numeric_cast<std::streamsize>(info->SizeOfData()));
            bool bRet = layer.ParseFromArray(buffer, static_cast<int>(info->SizeOfData()));
            delete[] buffer;
            if (!bRet)
            {
                throw armnn::ParseException("Failed to parse layer [" + info->name() + "]");
            }
        }

        if (info->new_tops())
        {
            //update the tops
            layer.set_top(0, info->top(0));
        }
        if (info->new_bottoms())
        {
            //update the bottoms
            layer.set_bottom(0, info->bottom(0));
        }

        auto it = ms_CaffeLayerNameToParsingFunctions.find(layer.type());
        if (it == ms_CaffeLayerNameToParsingFunctions.end())
        {
            throw armnn::ParseException("Unsupported layer type '" + layer.type() + "'");
        }
        auto func = it->second;
        (this->*func)(layer);
    }
    ifs.close();

    // Add ArmNN output layers connected to each requested output
    for (const std::string& requestedOutput : m_RequestedOutputs)
    {
        armnn::IOutputSlot& outputSlot = GetArmnnOutputSlotForCaffeTop(requestedOutput);

        const armnn::LayerBindingId outputId = armnn::numeric_cast<armnn::LayerBindingId>(
            m_NetworkOutputsBindingInfo.size());
        armnn::IConnectableLayer* const outputLayer = m_Network->AddOutputLayer(outputId, requestedOutput.c_str());
        outputSlot.Connect(outputLayer->GetInputSlot(0));

        TrackOutputBinding(outputLayer, outputId, outputLayer->GetInputSlot(0).GetConnection()->GetTensorInfo());
    }

    Cleanup();

    return move(m_Network);
}
