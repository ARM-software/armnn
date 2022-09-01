# Keyword Spotting with PyArmNN

This sample application guides the user to perform Keyword Spotting (KWS) with PyArmNN API.

## Prerequisites

### PyArmNN

Before proceeding to the next steps, make sure that you have successfully installed the newest version of PyArmNN on your system by following the instructions in the README of the PyArmNN root directory.

You can verify that PyArmNN library is installed and check PyArmNN version using:

```bash
$ pip show pyarmnn
```

You can also verify it by running the following and getting output similar to below:

```bash
$ python -c "import pyarmnn as ann;print(ann.GetVersion())"
'32.0.0'
```

### Dependencies

Install the PortAudio package:

```bash
$ sudo apt-get install libsndfile1 libportaudio2
```

Install the required Python modules: 

```bash
$ pip install -r requirements.txt
```

### Model

The model we are using is the [DS CNN Large](https://github.com/ARM-software/ML-zoo/raw/68b5fbc77ed28e67b2efc915997ea4477c1d9d5b/models/keyword_spotting/ds_cnn_large/tflite_clustered_int8/) which can be found in the [Arm Model Zoo repository](
https://github.com/ARM-software/ML-zoo/tree/master/models).

A small selection of suitable wav files containing keywords can be found [here](https://git.mlplatform.org/ml/ethos-u/ml-embedded-evaluation-kit.git/plain/resources/kws/samples/).

Labels for this model are defined within run_audio_classification.py.

## Performing Keyword Spotting

### Processing Audio Files

Please ensure that your audio file has a sampling rate of 16000Hz.

To run KWS on an audio file, use the following command:

```bash
$ python run_audio_classification.py --audio_file_path <path/to/your_audio> --model_file_path <path/to/your_model> 
```

You may also add the optional flags:

* `--preferred_backends`

  * Takes the preferred backends in preference order, separated by whitespace. For example, passing in "CpuAcc CpuRef" will be read as list ["CpuAcc", "CpuRef"] (defaults to this list)

    * CpuAcc represents the CPU backend

    * GpuAcc represents the GPU backend

    * CpuRef represents the CPU reference kernels

* `--help` prints all available options to screen


### Processing Audio Streams

To run KWS on a live audio stream, use the following command:

```bash
$ python run_audio_classification.py --model_file_path <path/to/your_model> --duration (optional)
```
You will be prompted to select an input microphone and inference will commence
after 3 seconds.


You may also add the following optional flag in addition to those for run_audio_file.py:

* `--duration`

  * Integer number of seconds to perform inference. Default is to continue indefinitely.

## Application Overview

1. [Initialization](#initialization)

2. [Creating a network](#creating-a-network)

3. [Keyword Spotting Pipeline](#keyword-spotting-pipeline)

### Initialization

The application parses the supplied user arguments and loads the audio file or stream in chunks through the `capture_audio()` method which accepts sampling criteria as an `AudioCaptureParams` tuple.

With KWS from an audio file, the application will create a generator object to yield blocks of audio data from the file with a minimum sample size defined in AudioCaptureParams. 

MFCC features are extracted from each block based on criteria defined in the `MFCCParams` tuple. These extracted features constitute the input tensors for the model.

To interpret the inference result of the loaded network; the application passes the label dictionary defined in run_audio_classification.py to a decoder and displays the result.

### Creating a network

A PyArmNN application must import a graph from file using an appropriate parser. Arm NN provides parsers for various model file types, including TFLite and ONNX. These parsers are libraries for loading neural networks of various formats into the Arm NN runtime.

Arm NN supports optimized execution on multiple CPU, GPU, and Ethos-N devices. Before executing a graph, the application must select the appropriate device context by using `IRuntime()` to create a runtime context with default options. We can optimize the imported graph by specifying a list of backends in order of preference and implementing backend-specific optimizations, identified by a unique string, for example CpuAcc, GpuAcc, CpuRef represent the accelerated CPU and GPU backends and the CPU reference kernels respectively.

Arm NN splits the entire graph into subgraphs based on these backends. Each subgraph is then optimized, and the corresponding subgraph in the original graph is substituted with its optimized version.

The `Optimize()` function optimizes the graph for inference, then `LoadNetwork()` loads the optimized network onto the compute device. The `LoadNetwork()` function also creates the backend-specific workloads for the layers and a backend-specific workload factory.

Parsers extract the input information for the network. The `GetSubgraphInputTensorNames()` function extracts all the input names and the `GetNetworkInputBindingInfo()` function obtains the input binding information of the graph. The input binding information contains all the essential information about the input. This information is a tuple consisting of integer identifiers for bindable layers and tensor information (data type, quantization info, dimension count, total elements).

Similarly, we can get the output binding information for an output layer by using the parser to retrieve output tensor names and calling the `GetNetworkOutputBindingInfo()` function

For this application, the main point of contact with PyArmNN is through the `ArmnnNetworkExecutor` class, which will handle the network creation step for you.

```python
# common/network_executor.py
# The provided kws model is in .tflite format so we use TfLiteParser() to import the graph
if ext == '.tflite':
    parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile(model_file)
...
# Optimize the network for the list of preferred backends
opt_network, messages = ann.Optimize(
    network, preferred_backends, self.runtime.GetDeviceSpec(), ann.OptimizerOptions()
    )
# Load the optimized network onto the runtime device
self.network_id, _ = self.runtime.LoadNetwork(opt_network)
# Get the input and output binding information
self.input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
self.output_binding_info = parser.GetNetworkOutputBindingInfo(graph_id, output_name)
```

### Keyword Spotting pipeline


Mel-frequency Cepstral Coefficients (MFCCs, [see Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)) are extracted based on criteria defined in the MFCCParams tuple and associated`MFCC Class`.
MFCCs are the result of computing the dot product of the Discrete Cosine Transform (DCT) Matrix and the log of the Mel energy.

The `MFCC` class is used in conjunction with the `AudioPreProcessor` class to extract and process MFCC features from a given audio frame. 


After all the MFCCs needed for an inference have been extracted from the audio data they constitute the input tensors that will be classified by an `ArmnnNetworkExecutor`object.

```python
# mfcc.py
# Extract MFCC features from audio_data
audio_data.resize(self._frame_len_padded)
spec = self.spectrum_calc(audio_data)
mel_energy = np.dot(self._np_mel_bank.astype(np.float32),
                    np.transpose(spec).astype(np.float32))
log_mel_energy = self.log_mel(mel_energy)
mfcc_feats = np.dot(self._dct_matrix, log_mel_energy)


```python
# audio_utils.py
# Quantize the input data and create input tensors with PyArmNN
input_tensor = quantize_input(input_tensor, input_binding_info)
input_tensors = ann.make_input_tensors([input_binding_info], [input_data])
```

Note: `ArmnnNetworkExecutor` has already created the output tensors for you.

After creating the workload tensors, the compute device performs inference for the loaded network by using the `EnqueueWorkload()` function of the runtime context. Calling the `workload_tensors_to_ndarray()` function obtains the inference results as a list of ndarrays.

```python
# common/network_executor.py
status = runtime.EnqueueWorkload(net_id, input_tensors, self.output_tensors)
self.output_result = ann.workload_tensors_to_ndarray(self.output_tensors)
```

The output from the inference must be decoded to obtain the recognised classification. A simple greedy decoder classifies the results by taking the highest element of the output as a key for the labels dictionary. The value returned is a keyword or unknown/silence which is appended to a list along with the calculated probability. The list elements are displayed on the console if they exceed the threshold value specified in main().


## Next steps

Having now gained a solid understanding of performing keyword spotting with PyArmNN, you are able to take control and create your own application. We suggest to first implement your own network, which can be done by updating the parameters of `AudioCaptureParams` and `MFCC_Params` to match your custom model. The `ArmnnNetworkExecutor` class will handle the network optimisation and loading for you.

An important factor in improving accuracy of the generated output is providing cleaner data to the network. This can be done by including additional preprocessing steps such as noise reduction of your audio data.
