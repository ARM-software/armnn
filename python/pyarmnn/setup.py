#!/usr/bin/env python3
# Copyright © 2020 Arm Ltd. All rights reserved.
# Copyright © 2020 NXP and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
"""Python bindings for Arm NN

PyArmNN is a python extension for Arm NN SDK providing an interface similar to Arm NN C++ API.
"""
__version__ = None
__arm_ml_version__ = None

import logging
import os
import sys
import subprocess
from functools import lru_cache
from pathlib import Path
from itertools import chain

from setuptools import setup
from distutils.core import Extension
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext

logger = logging.Logger(__name__)

DOCLINES = __doc__.split("\n")
LIB_ENV_NAME = "ARMNN_LIB"
INCLUDE_ENV_NAME = "ARMNN_INCLUDE"


def check_armnn_version(*args):
    pass

__current_dir = os.path.dirname(os.path.realpath(__file__))

exec(open(os.path.join(__current_dir, 'src', 'pyarmnn', '_version.py'), encoding="utf-8").read())


class ExtensionPriorityBuilder(build_py):
    """Runs extension builder before other stages. Otherwise generated files are not included to the distribution.
    """

    def run(self):
        self.run_command('build_ext')
        return super().run()


class ArmnnVersionCheckerExtBuilder(build_ext):
    """Builds an extension (i.e. wrapper). Additionally checks for version.
    """

    def __init__(self, dist):
        super().__init__(dist)
        self.failed_ext = []

    def build_extension(self, ext):
        if ext.optional:
            try:
                super().build_extension(ext)
            except Exception as err:
                self.failed_ext.append(ext)
                logger.warning('Failed to build extension %s. \n %s', ext.name, str(err))
        else:
            super().build_extension(ext)
            if ext.name == 'pyarmnn._generated._pyarmnn_version':
                sys.path.append(os.path.abspath(os.path.join(self.build_lib, str(Path(ext._file_name).parent))))
                from _pyarmnn_version import GetVersion
                check_armnn_version(GetVersion(), __arm_ml_version__)

    def copy_extensions_to_source(self):

        for ext in self.failed_ext:
            self.extensions.remove(ext)
        super().copy_extensions_to_source()


def linux_gcc_name():
    """Returns the name of the `gcc` compiler. Might happen that we are cross-compiling and the
    compiler has a longer name.

    Args:
        None

    Returns:
        str: Name of the `gcc` compiler or None
    """
    cc_env = os.getenv('CC')
    if cc_env is not None:
        if subprocess.Popen([cc_env, "--version"], stdout=subprocess.DEVNULL):
            return cc_env
    return "gcc" if subprocess.Popen(["gcc", "--version"], stdout=subprocess.DEVNULL) else None


def linux_gcc_lib_search(gcc_compiler_name: str = linux_gcc_name()):
    """Calls the `gcc` to get linker default system paths.

    Args:
        gcc_compiler_name(str): Name of the GCC compiler

    Returns:
        list: A list of paths.

    Raises:
        RuntimeError: If unable to find GCC.
    """
    if gcc_compiler_name is None:
        raise RuntimeError("Unable to find gcc compiler")
    cmd1 = subprocess.Popen([gcc_compiler_name, "--print-search-dirs"], stdout=subprocess.PIPE)
    cmd2 = subprocess.Popen(["grep", "libraries"], stdin=cmd1.stdout,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cmd1.stdout.close()
    out, _ = cmd2.communicate()
    out = out.decode("utf-8").split('=')
    return tuple(out[1].split(':')) if len(out) > 0 else None


def find_includes(armnn_include_env: str = INCLUDE_ENV_NAME):
    """Searches for ArmNN includes.

    Args:
        armnn_include_env(str): Environmental variable to use as path.

    Returns:
        list: A list of paths to include.
    """

    # split multiple paths
    global armnn_include_path
    armnn_include_path_raw = os.getenv(armnn_include_env)
    if not armnn_include_path_raw == None:
        armnn_include_path = armnn_include_path_raw.split(":")

    # validate input paths
    armnn_include_path_result = []
    for path in armnn_include_path:
        if path is not None and os.path.exists(path):
            armnn_include_path_result = armnn_include_path_result + [path]


    # if none exist revert to default
    if len(armnn_include_path_result) == 0:
        armnn_include_path_result = ['/usr/local/include', '/usr/include']
    return armnn_include_path_result



@lru_cache(maxsize=1)
def find_armnn(lib_name: str,
               optional: bool = False,
               armnn_libs_env: str = LIB_ENV_NAME,
               default_lib_search: tuple = linux_gcc_lib_search()):
    """Searches for ArmNN installation on the local machine.

    Args:
        lib_name(str): Lib name to find.
        optional(bool): Do not fail if optional. Default is False - fail if library was not found.
        armnn_libs_env(str): Custom environment variable pointing to ArmNN libraries location, default is 'ARMNN_LIBS'
        default_lib_search(tuple): list of paths to search for ArmNN if not found within path provided by 'ARMNN_LIBS'
                            env variable
    Returns:
        tuple: Contains name of the armnn libs, paths to the libs.

    Raises:
        RuntimeError: If armnn libs are not found.
    """
    armnn_lib_path = os.getenv(armnn_libs_env)
    lib_search = [armnn_lib_path] if armnn_lib_path is not None else default_lib_search
    armnn_libs = dict(map(lambda path: (':{}'.format(path.name), path),
                          chain.from_iterable(map(lambda lib_path: Path(lib_path).glob(lib_name),
                                                  lib_search))))
    if not optional and len(armnn_libs) == 0:
        raise RuntimeError("""ArmNN library {} was not found in {}. Please install ArmNN to one of the standard
                           locations or set correct ARMNN_INCLUDE and ARMNN_LIB env variables.""".format(lib_name,
                                                                                                         lib_search))
    if optional and len(armnn_libs) == 0:
        logger.warning("""Optional parser library %s was not found in %s and will not be installed.""", lib_name,
                                                                                                        lib_search)

    # gives back tuple of names of the libs, set of unique libs locations and includes.
    return list(armnn_libs.keys()), list(set(
        map(lambda path: str(path.absolute().parent), armnn_libs.values())))


class LazyArmnnFinderExtension(Extension):
    """Derived from `Extension` this class adds ArmNN libraries search on the user's machine.
    SWIG options and compilation flags are updated with relevant ArmNN libraries files locations (-L) and headers (-I).

    Search for ArmNN is executed only when attributes include_dirs, library_dirs, runtime_library_dirs, libraries or
    swig_opts are queried.

    """

    def __init__(self, name, sources, armnn_libs, include_dirs=None, define_macros=None, undef_macros=None,
                 library_dirs=None,
                 libraries=None, runtime_library_dirs=None, extra_objects=None, extra_compile_args=None,
                 extra_link_args=None, export_symbols=None, language=None, optional=None, **kw):
        self._include_dirs = None
        self._library_dirs = None
        self._runtime_library_dirs = None
        self._armnn_libs = armnn_libs
        self._optional = False if optional is None else optional

        super().__init__(name=name, sources=sources, include_dirs=include_dirs, define_macros=define_macros,
                         undef_macros=undef_macros, library_dirs=library_dirs, libraries=libraries,
                         runtime_library_dirs=runtime_library_dirs, extra_objects=extra_objects,
                         extra_compile_args=extra_compile_args, extra_link_args=extra_link_args,
                         export_symbols=export_symbols, language=language, optional=optional, **kw)

    @property
    def include_dirs(self):
        return self._include_dirs + find_includes()

    @include_dirs.setter
    def include_dirs(self, include_dirs):
        self._include_dirs = include_dirs

    @property
    def library_dirs(self):
        library_dirs = self._library_dirs
        for lib in self._armnn_libs:
            _, lib_path = find_armnn(lib, self._optional)
            library_dirs = library_dirs + lib_path

        return library_dirs

    @library_dirs.setter
    def library_dirs(self, library_dirs):
        self._library_dirs = library_dirs

    @property
    def runtime_library_dirs(self):
        library_dirs = self._runtime_library_dirs
        for lib in self._armnn_libs:
            _, lib_path = find_armnn(lib, self._optional)
            library_dirs = library_dirs + lib_path

        return library_dirs

    @runtime_library_dirs.setter
    def runtime_library_dirs(self, runtime_library_dirs):
        self._runtime_library_dirs = runtime_library_dirs

    @property
    def libraries(self):
        libraries = self._libraries
        for lib in self._armnn_libs:
            lib_names, _ = find_armnn(lib, self._optional)
            libraries = libraries + lib_names

        return libraries

    @libraries.setter
    def libraries(self, libraries):
        self._libraries = libraries

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.name.__hash__()


if __name__ == '__main__':
    # mandatory extensions
    pyarmnn_module = LazyArmnnFinderExtension('pyarmnn._generated._pyarmnn',
                                              sources=['src/pyarmnn/_generated/armnn_wrap.cpp'],
                                              extra_compile_args=['-std=c++14'],
                                              language='c++',
                                              armnn_libs=['libarmnn.so'],
                                              optional=False
                                              )
    pyarmnn_v_module = LazyArmnnFinderExtension('pyarmnn._generated._pyarmnn_version',
                                                sources=['src/pyarmnn/_generated/armnn_version_wrap.cpp'],
                                                extra_compile_args=['-std=c++14'],
                                                language='c++',
                                                armnn_libs=['libarmnn.so'],
                                                optional=False
                                                )
    extensions_to_build = [pyarmnn_v_module, pyarmnn_module]


    # optional extensions
    def add_parsers_ext(name: str, ext_list: list):
        pyarmnn_optional_module = LazyArmnnFinderExtension('pyarmnn._generated._pyarmnn_{}'.format(name.lower()),
                                                           sources=['src/pyarmnn/_generated/armnn_{}_wrap.cpp'.format(
                                                               name.lower())],
                                                           extra_compile_args=['-std=c++14'],
                                                           language='c++',
                                                           armnn_libs=['libarmnn.so', 'libarmnn{}.so'.format(name)],
                                                           optional=True
                                                           )
        ext_list.append(pyarmnn_optional_module)


    add_parsers_ext('OnnxParser', extensions_to_build)
    add_parsers_ext('TfLiteParser', extensions_to_build)
    add_parsers_ext('Deserializer', extensions_to_build)

    setup(
        name='pyarmnn',
        version=__version__,
        author='Arm Ltd, NXP Semiconductors',
        author_email='support@linaro.org',
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        url='https://mlplatform.org/',
        license='MIT',
        keywords='armnn neural network machine learning',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        package_dir={'': 'src'},
        packages=[
            'pyarmnn',
            'pyarmnn._generated',
            'pyarmnn._quantization',
            'pyarmnn._tensor',
            'pyarmnn._utilities'
        ],
        data_files=[('', ['LICENSE'])],
        python_requires='>=3.5',
        install_requires=['numpy'],
        cmdclass={
            'build_py': ExtensionPriorityBuilder,
            'build_ext': ArmnnVersionCheckerExtBuilder
        },
        ext_modules=extensions_to_build
    )
