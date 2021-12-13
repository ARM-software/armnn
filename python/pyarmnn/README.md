# PyArmNN

PyArmNN is a python extension for [Arm NN SDK](https://developer.arm.com/ip-products/processors/machine-learning/arm-nn).
PyArmNN provides interface similar to Arm NN C++ Api.
Before you proceed with the project setup, you will need to checkout and build a corresponding Arm NN version.

PyArmNN is built around public headers from the armnn/include folder of Arm NN. PyArmNN does not implement any computation kernels itself, all operations are
delegated to the Arm NN library.

The [SWIG](http://www.swig.org/) project is used to generate the Arm NN python shadow classes and C wrapper.

The following diagram shows the conceptual architecture of this library:
![PyArmNN](docs/pyarmnn.png)

# Setup development environment

Before, proceeding to the next steps, make sure that:

1. You have Python 3.6+ installed system-side. The package is not compatible with older Python versions.
2. You have python3.6-dev installed system-side. This contains header files needed to build PyArmNN extension module.
3. In case you build Python from sources manually, make sure that the following libraries are installed and available in you system:
``python3.6-dev build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev``
4. Install SWIG 4.x. Only 3.x version is typically available in Linux package managers, so you will have to build it and install it from sources. It can be downloaded from the [SWIG project website](http://www.swig.org/download.html) or from [SWIG GitHub](https://github.com/swig/swig). To install it follow the guide on [SWIG GitHub](https://github.com/swig/swig/wiki/Getting-Started).

## Setup virtual environment

Now you can proceed with setting up workspace. It is recommended to create a python virtual environment, so you do not pollute your working folder:
```bash
python -m venv env
source env/bin/activate
```

You may run into missing python modules such as *wheel*. Make sure to install those using pip:
```bash
pip install wheel
```

## Build python distr

Python supports source and binary distribution packages.

Source distr contains setup.py script that is executed on the users machine during package installation.
When preparing binary distr (wheel), setup.py is executed on the build machine and the resulting package contains only the result
of the build (generated files and resources, test results etc).

In our case, PyArmNN depends on Arm NN installation. Thus, binary distr will be linked with
the local build machine libraries and runtime.

There are 2 ways to build the python packages. Either directly using the python scripts or using CMake.

### CMake build

The recommended aproach is to build PyArmNN together with Arm NN by adding the following options to your CMake command:
```
-DBUILD_PYTHON_SRC=1
-DBUILD_PYTHON_WHL=1
```
This will build either the source package or the wheel or both. Current project headers and build libraries will be used, so there is no need to provide them.

SWIG is required to generate the wrappers. If CMake did not find the executable during the configure step or it has found an older version, you may provide it manually:
```
-DSWIG_EXECUTABLE=<path_to_swig_executable>
```

After the build finishes, you will find the python packages in `<build_folder>/python/pyarmnn/dist`.

### Standalone build

PyArmNN can also be built using the provided python scripts only. The advantage of that is that you may use prebuilt Arm NN libraries and it is generally much faster if you do not want to build all the Arm NN libraries.

##### 1. Set environment:

*ARMNN_INCLUDE* and *ARMNN_LIB* are mandatory and should point to Arm NN includes and libraries against which you will be generating the wrappers. *SWIG_EXECUTABLE* should only be set if you have multiple versions of SWIG installed or you used a custom location for your installation:
```bash
$ export SWIG_EXECUTABLE=/full/path/to/swig/executable
$ export ARMNN_INCLUDE=/full/path/to/armnn/include:/full/path/to/armnn/profiling/common/include
$ export ARMNN_LIB=/path/to/libs
```

##### 2. Clean and build SWIG wrappers:

```bash
$ python setup.py clean --all
$ python swig_generate.py -v
$ python setup.py build_ext --inplace
```
This step will put all generated files under `./src/pyarmnn/_generated` folder and can be used repeatedly to re-generate the wrappers.

##### 4. Build the source package

```bash
$ python setup.py sdist
```
As the result you will get `./dist/pyarmnn-28.0.0.tar.gz` file. As you can see it is platform independent.

##### 5. Build the binary package

```bash
$ python setup.py bdist_wheel
```
As the result you will get something like `./dist/pyarmnn-28.0.0-cp36-cp36m-linux_x86_64.whl` file. As you can see it
 is platform dependent.

# PyArmNN installation

PyArmNN can be distributed as a source package or a binary package (wheel).

Binary package is platform dependent, the name of the package will indicate the platform it was built for, e.g.:

* Linux x86 64bit machine: pyarmnn-28.0.0-cp36-cp36m-*linux_x86_64*.whl
* Linux Aarch 64 bit machine: pyarmnn-28.0.0-cp36-cp36m-*linux_aarch64*.whl

The source package is platform independent but installation involves compilation of Arm NN python extension. You will need to have g++ compatible with C++ 14 standard and a python development library installed on the build machine.

Both of them, source and binary package, require the Arm NN library to be present on the target/build machine.

It is strongly suggested to work within a python virtual environment. The further steps assume that the virtual environment was created and activated before running PyArmNN installation commands.

PyArmNN also depends on the NumPy python library. It will be automatically downloaded and installed alongside PyArmNN. If your machine does not have access to Python pip repositories you might need to install NumPy in advance by following public instructions: https://scipy.org/install.html

## Installing from wheel

Make sure that Arm NN binaries and Arm NN dependencies are installed and can be found in one of the system default library locations. You can check default locations by executing the following command:
```bash
$ gcc --print-search-dirs
```
Install PyArmNN from binary by pointing to the wheel file:
```bash
$ pip install /path/to/pyarmnn-28.0.0-cp36-cp36m-linux_aarch64.whl
```

## Installing from source package

Alternatively, you can install from source. This is the more reliable way but requires a little more effort on the users part.

While installing from sources, you have the freedom of choosing Arm NN libraries location. Set environment variables *ARMNN_LIB* and *ARMNN_INCLUDE* to point to Arm NN libraries and headers.
If you want to use system default locations, just set *ARMNN_INCLUDE* to point to Arm NN headers.

```bash
$ export  ARMNN_LIB=/path/to/libs
$ export  ARMNN_INCLUDE=/full/path/to/armnn/include:/full/path/to/armnn/profiling/common/include
```

Install PyArmNN as follows:
```bash
$ pip install /path/to/pyarmnn-28.0.0.tar.gz
```

If PyArmNN installation script fails to find Arm NN libraries it will raise an error like this

`RuntimeError: ArmNN library was not found in ('/usr/lib/gcc/aarch64-linux-gnu/8/', <...> ,'/lib/', '/usr/lib/'). Please install ArmNN to one of the standard locations or set correct ARMNN_INCLUDE and ARMNN_LIB env variables.`

You can now verify that PyArmNN library is installed and check PyArmNN version using:
```bash
$ pip show pyarmnn
```
You can also verify it by running the following and getting output similar to below:
```bash
$ python -c "import pyarmnn as ann;print(ann.GetVersion())"
'28.0.0'
```

# PyArmNN API overview

#### Getting started
The easiest way to begin using PyArmNN is by using the Parsers. We will demonstrate how to use them below:

Create a parser object and load your model file.
```python
import pyarmnn as ann
import imageio

# An ONNX parser also exists.
parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile('./model.tflite')
```

Get the input binding information by using the name of the input layer.
```python
input_binding_info = parser.GetNetworkInputBindingInfo(0, 'model/input')

# Create a runtime object that will perform inference.
options = ann.CreationOptions()
runtime = ann.IRuntime(options)
```
Choose preferred backends for execution and optimize the network.
```python
# Backend choices earlier in the list have higher preference.
preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

# Load the optimized network into the runtime.
net_id, _ = runtime.LoadNetwork(opt_network)
```
Make workload tensors using input and output binding information.
```python
# Load an image and create an inputTensor for inference.
img = imageio.imread('./image.png')
input_tensors = ann.make_input_tensors([input_binding_info], [img])

# Get output binding information for an output layer by using the layer name.
output_binding_info = parser.GetNetworkOutputBindingInfo(0, 'model/output')
output_tensors = ann.make_output_tensors([output_binding_info])
```

Perform inference and get the results back into a numpy array.
```python
runtime.EnqueueWorkload(0, input_tensors, output_tensors)

results = ann.workload_tensors_to_ndarray(output_tensors)
print(results)
```

#### Examples

To further explore PyArmNN API there are several examples provided in the `/examples` folder for you to explore.

##### Image Classification

This sample application performs image classification on an image and outputs the <i>Top N</i> results, listing the classes and probabilities associated with the classified image. All resources are downloaded during execution, so if you do not have access to the internet, you may need to download these manually.

Sample scripts are provided for performing image classification with TFLite and ONNX models with `tflite_mobilenetv1_quantized.py` and `onnx_mobilenetv2.py`.

##### Object Detection

This sample application guides the user and shows how to perform object detection using PyArmNN API. By taking a model and video file or camera feed as input, and running inference on each frame, we are able to interpret the output to draw bounding boxes around detected objects and overlay the corresponding labels and confidence scores.

Sample scripts are provided for performing object detection from video file and video stream with `run_video_file.py` and `run_video_stream.py`.


## Tox for automation

To make things easier *tox* is available for automating individual tasks or running multiple commands at once such as generating wrappers, running unit tests using multiple python versions or generating documentation. To run it use:

```bash
$ tox <task_name>
```

See *tox.ini* for the list of tasks. You may also modify it for your own purposes. To dive deeper into tox read through https://tox.readthedocs.io/en/latest/

## Running unit-tests

Download resources required to run unit tests by executing the script in the scripts folder:

```
$ python ./scripts/download_test_resources.py
```

The script will download an archive from the Linaro server and extract it. A folder `test/testdata/shared` will be created. Execute `pytest` from the project root dir:
```bash
$ python -m pytest test/ -v
```
or run tox which will do both:
```bash
$ tox
```
