"""
Build script for TurboQuant XPU kernels (Intel ARC).

Usage:
    python setup_xpu.py build_ext --inplace
"""

import os
import sys
import subprocess
import setuptools
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

# Robust oneAPI detection for Windows
if sys.platform == "win32":
    if "ONEAPI_ROOT" not in os.environ:
        potential_path = r"C:\Program Files (x86)\Intel\oneAPI"
        if os.path.exists(potential_path):
            os.environ["ONEAPI_ROOT"] = potential_path
            print(f"Detected oneAPI at {potential_path}")
        else:
            print("WARNING: ONEAPI_ROOT not found and default path does not exist.")

    # Ensure compiler is in path
    oneapi_root = os.environ.get("ONEAPI_ROOT", "").strip()
    if oneapi_root:
        compiler_path = os.path.join(oneapi_root, "compiler", "latest", "bin")
        if compiler_path not in os.environ["PATH"]:
            os.environ["PATH"] = compiler_path + os.pathsep + os.environ["PATH"]

try:
    from torch.utils.cpp_extension import SyclExtension
    import torch
    import torch.utils.cpp_extension
    import shutil
    
    if torch.utils.cpp_extension.SYCL_HOME is None:
        icx_path = shutil.which('icx')
        if icx_path:
            torch.utils.cpp_extension.SYCL_HOME = os.path.dirname(os.path.dirname(os.path.realpath(icx_path)))
            print(f"Patched SYCL_HOME to {torch.utils.cpp_extension.SYCL_HOME}")

    if sys.platform == "win32":
        os.environ['CXX'] = 'icx'
        os.environ['CC'] = 'icx'
        os.environ['TORCH_DONT_CHECK_COMPILER_ABI'] = '1'

    def sycl_flags():
        if sys.platform == "win32":
            return ["/O2", "/std:c++17", "-fsycl", "/fp:fast"]
        else:
            return ["-O3", "-std=c++17", "-fsycl", "-ffast-math"]

    csrc_dir = os.path.join(os.path.dirname(__file__), 'turboquant', 'csrc')

    # Add torch lib path for linking
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    sycl_lib_path = os.path.join(os.environ.get('ONEAPI_ROOT', 'C:\\Program Files (x86)\\Intel\\oneAPI').strip(), 'compiler', 'latest', 'lib')
    
    ext_modules = [
        SyclExtension(
            name='turboquant.xpu_rotor_fused',
            sources=[os.path.join(csrc_dir, 'rotor_fused_kernel_xpu.cpp')],
            include_dirs=[],
            library_dirs=[torch_lib_path, sycl_lib_path],
            extra_compile_args={"cxx": sycl_flags()},
            extra_link_args=["-fsycl", f"/LIBPATH:{sycl_lib_path}"]
        ),
        SyclExtension(
            name='turboquant.xpu_qjl_score',
            sources=[os.path.join(csrc_dir, 'qjl_score_kernel_xpu.cpp')],
            include_dirs=[],
            library_dirs=[torch_lib_path, sycl_lib_path],
            extra_compile_args={"cxx": sycl_flags()},
            extra_link_args=["-fsycl", f"/LIBPATH:{sycl_lib_path}"]
        ),
        SyclExtension(
            name='turboquant.xpu_qjl_quant',
            sources=[os.path.join(csrc_dir, 'qjl_quant_kernel_xpu.cpp')],
            include_dirs=[],
            library_dirs=[torch_lib_path, sycl_lib_path],
            extra_compile_args={"cxx": sycl_flags()},
            extra_link_args=["-fsycl", f"/LIBPATH:{sycl_lib_path}"]
        ),
        SyclExtension(
            name='turboquant.xpu_qjl_gqa_score',
            sources=[os.path.join(csrc_dir, 'qjl_gqa_score_kernel_xpu.cpp')],
            include_dirs=[],
            library_dirs=[torch_lib_path, sycl_lib_path],
            extra_compile_args={"cxx": sycl_flags()},
            extra_link_args=["-fsycl", f"/LIBPATH:{sycl_lib_path}"]
        ),
        SyclExtension(
            name='turboquant.xpu_quantization',
            sources=[os.path.join(csrc_dir, 'quantization_xpu.cpp')],
            include_dirs=[],
            library_dirs=[torch_lib_path, sycl_lib_path],
            extra_compile_args={"cxx": sycl_flags()},
            extra_link_args=["-fsycl"]
        ),
        SyclExtension(
            name='turboquant.xpu_iso_planar',
            sources=[os.path.join(csrc_dir, 'iso_planar_kernels_xpu.cpp')],
            include_dirs=[],
            library_dirs=[torch_lib_path, sycl_lib_path],
            extra_compile_args={"cxx": sycl_flags()},
            extra_link_args=["-fsycl", f"/LIBPATH:{sycl_lib_path}"]
        ),
    ]
    class XPUBuildExtension(BuildExtension):
        def build_extensions(self):
            if sys.platform == "win32":
                self.compiler.cc = 'icx'
                self.compiler.cxx = 'icx'
                if hasattr(self.compiler, 'linker_so'):
                    self.compiler.linker_so[0] = 'icx'
                if hasattr(self.compiler, 'linker_exe'):
                    self.compiler.linker_exe[0] = 'icx'
                
                original_spawn = self.compiler.spawn
                def spawn(cmd, **kwargs):
                    executable = os.path.basename(cmd[0]).lower()
                    if executable.startswith('cl') or executable.startswith('link'):
                        is_linking = executable.startswith('link')
                        new_cmd = ['icx']
                        if is_linking:
                            new_cmd.append('-fsycl')
                            compiler_args = []
                            linker_args = ['/link']
                            for arg in cmd[1:]:
                                # Linker specific flags
                                if any(arg.startswith(prefix) for prefix in ['/LIBPATH:', '/OUT:', '/IMPLIB:', '/nologo', '/INCREMENTAL', '/LTCG', '/DLL', '/MANIFEST', '/MACHINE', '/DYNAMICBASE', '/NXCOMPAT']):
                                    linker_args.append(arg)
                                elif arg.endswith('.lib'):
                                    linker_args.append(arg)
                                elif arg.startswith('-fsycl'):
                                    continue # already added
                                else:
                                    compiler_args.append(arg)
                            new_cmd.extend(compiler_args)
                            new_cmd.extend(linker_args)
                        else:
                            # Compile phase
                            for arg in cmd[1:]:
                                if (arg.startswith('-I') or arg.startswith('/I')) and ('Microsoft Visual Studio' in arg or 'Windows Kits' in arg):
                                    new_cmd.append('-imsvc')
                                    new_cmd.append(arg[2:])
                                else:
                                    new_cmd.append(arg)
                        cmd = new_cmd
                    print(f"RUNNING: {' '.join(cmd)}")
                    return original_spawn(cmd, **kwargs)
                self.compiler.spawn = spawn
            super().build_extensions()

    cmdclass = {'build_ext': XPUBuildExtension.with_options(use_ninja=False)}
    print("XPU extensions will be built using icx without Ninja.")
except ImportError as e:
    print(f"WARNING: failed to import SyclExtension or torch: {e}")
    ext_modules = []
    cmdclass = {}

setup(
    name='turboquant_xpu',
    version='0.2.0',
    description='TurboQuant: XPU Kernels',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
    ],
)
