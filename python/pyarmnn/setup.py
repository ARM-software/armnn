# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from itertools import chain

from setuptools import setup
from distutils.core import Extension
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext

logger = logging.Logger(__name__)

__version__ = None
__arm_ml_version__ = None


def check_armnn_version(*args):
    pass


exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'pyarmnn', '_version.py')).read())


class ExtensionPriorityBuilder(build_py):
    """
    Runs extension builder before other stages. Otherwise generated files are not included to the distribution.
    """

    def run(self):
        self.run_command('build_ext')
        return super().run()


class ArmnnVersionCheckerExtBuilder(build_ext):

    def __init__(self, dist):
        super().__init__(dist)
        self.failed_ext = []

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as err:
            self.failed_ext.append(ext)
            logger.warning('Failed to build extension %s. \n %s', ext.name, str(err))

        if ext.name == 'pyarmnn._generated._pyarmnn_version':
            sys.path.append(os.path.abspath(os.path.join(self.build_lib, str(Path(ext._file_name).parent))))
            from _pyarmnn_version import GetVersion
            check_armnn_version(GetVersion(), __arm_ml_version__)

    def copy_extensions_to_source(self):

        for ext in self.failed_ext:
            self.extensions.remove(ext)
        super().copy_extensions_to_source()


def linux_gcc_lib_search():
    """
    Calls the `gcc` to get linker default system paths.
    Returns:
        list of paths
    """
    cmd = 'gcc --print-search-dirs | grep libraries'
    cmd_res = os.popen(cmd).read()
    cmd_res = cmd_res.split('=')
    if len(cmd_res) > 1:
        return tuple(cmd_res[1].split(':'))
    return None


def find_includes(armnn_include_env: str = 'ARMNN_INCLUDE'):
    armnn_include_path = os.getenv(armnn_include_env, '')
    return [armnn_include_path] if armnn_include_path else ['/usr/local/include', '/usr/include']


@lru_cache(maxsize=1)
def find_armnn(lib_name: str,
               optional: bool = False,
               armnn_libs_env: str = 'ARMNN_LIB',
               default_lib_search: tuple = linux_gcc_lib_search()):
    """
    Searches for ArmNN installation on the local machine.

    Args:
        lib_name: lib name to find
        optional: Do not fail if optional. Default is False - fail if library was not found.
        armnn_include_env: custom environment variable pointing to ArmNN headers, default is 'ARMNN_INCLUDE'
        armnn_libs_env: custom environment variable pointing to ArmNN libraries location, default is 'ARMNN_LIBS'
        default_lib_search: list of paths to search for ArmNN if not found within path provided by 'ARMNN_LIBS'
                            env variable

    Returns:
        tuple containing name of the armnn libs, paths to the libs
    """

    armnn_lib_path = os.getenv(armnn_libs_env, "")

    lib_search = [armnn_lib_path] if armnn_lib_path else default_lib_search

    armnn_libs = dict(map(lambda path: (':{}'.format(path.name), path),
                          chain.from_iterable(map(lambda lib_path: Path(lib_path).glob(lib_name),
                                                  lib_search))))
    if not optional and len(armnn_libs) == 0:
        raise RuntimeError("""ArmNN library {} was not found in {}. Please install ArmNN to one of the standard
                           locations or set correct ARMNN_INCLUDE and ARMNN_LIB env variables.""".format(lib_name,
                                                                                                         lib_search))

    # gives back tuple of names of the libs, set of unique libs locations and includes.
    return list(armnn_libs.keys()), list(set(
        map(lambda path: str(path.absolute().parent), armnn_libs.values())))


class LazyArmnnFinderExtension(Extension):
    """
    Derived from `Extension` this class adds ArmNN libraries search on the user's machine.
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
        # self.__swig_opts = None
        super().__init__(name, sources, include_dirs, define_macros, undef_macros, library_dirs, libraries,
                         runtime_library_dirs, extra_objects, extra_compile_args, extra_link_args, export_symbols,
                         language, optional, **kw)

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
            _, lib_path = find_armnn(lib)
            library_dirs = library_dirs + lib_path

        return library_dirs

    @library_dirs.setter
    def library_dirs(self, library_dirs):
        self._library_dirs = library_dirs

    @property
    def runtime_library_dirs(self):
        library_dirs = self._runtime_library_dirs
        for lib in self._armnn_libs:
            _, lib_path = find_armnn(lib)
            library_dirs = library_dirs + lib_path

        return library_dirs

    @runtime_library_dirs.setter
    def runtime_library_dirs(self, runtime_library_dirs):
        self._runtime_library_dirs = runtime_library_dirs

    @property
    def libraries(self):
        libraries = self._libraries
        for lib in self._armnn_libs:
            lib_names, _ = find_armnn(lib)
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
                                              armnn_libs=['libarmnn.so']
                                              )
    pyarmnn_v_module = LazyArmnnFinderExtension('pyarmnn._generated._pyarmnn_version',
                                                sources=['src/pyarmnn/_generated/armnn_version_wrap.cpp'],
                                                extra_compile_args=['-std=c++14'],
                                                language='c++',
                                                armnn_libs=['libarmnn.so']
                                                )
    extensions_to_build = [pyarmnn_v_module, pyarmnn_module]


    # optional extensions
    def add_parsers_ext(name: str, ext_list: list):
        pyarmnn_optional_module = LazyArmnnFinderExtension('pyarmnn._generated._pyarmnn_{}'.format(name.lower()),
                                                           sources=['src/pyarmnn/_generated/armnn_{}_wrap.cpp'.format(
                                                               name.lower())],
                                                           extra_compile_args=['-std=c++14'],
                                                           language='c++',
                                                           armnn_libs=['libarmnn.so', 'libarmnn{}.so'.format(name)]
                                                           )
        ext_list.append(pyarmnn_optional_module)


    add_parsers_ext('CaffeParser', extensions_to_build)
    add_parsers_ext('OnnxParser', extensions_to_build)
    add_parsers_ext('TfParser', extensions_to_build)
    add_parsers_ext('TfLiteParser', extensions_to_build)

    setup(
        name='pyarmnn',
        version=__version__,
        author='Arm ltd',
        author_email='support@linaro.org',
        description='Arm NN python wrapper',
        url='https://www.arm.com',
        license='MIT',
        package_dir={'': 'src'},
        packages=[
            'pyarmnn',
            'pyarmnn._generated',
            'pyarmnn._quantization',
            'pyarmnn._tensor',
            'pyarmnn._utilities'
        ],
        python_requires='>=3.5',
        install_requires=['numpy'],
        cmdclass={'build_py': ExtensionPriorityBuilder, 'build_ext': ArmnnVersionCheckerExtBuilder},
        ext_modules=extensions_to_build
    )
