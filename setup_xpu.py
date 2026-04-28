"""
Build script for TurboQuant XPU kernels (Intel ARC).

Usage:
    python setup_xpu.py build_ext --inplace
"""

from setuptools import setup, find_packages
import os
import sys

try:
    from torch.utils.cpp_extension import BuildExtension, SyclExtension
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

    ext_modules = [
        SyclExtension(
            name='turboquant.xpu_rotor_fused',
            sources=[os.path.join(csrc_dir, 'rotor_fused_kernel_xpu.cpp')],
            extra_compile_args={"cxx": sycl_flags()}
        ),
        SyclExtension(
            name='turboquant.xpu_qjl_score',
            sources=[os.path.join(csrc_dir, 'qjl_score_kernel_xpu.cpp')],
            extra_compile_args={"cxx": sycl_flags()}
        ),
        SyclExtension(
            name='turboquant.xpu_qjl_quant',
            sources=[os.path.join(csrc_dir, 'qjl_quant_kernel_xpu.cpp')],
            extra_compile_args={"cxx": sycl_flags()}
        ),
        SyclExtension(
            name='turboquant.xpu_qjl_gqa_score',
            sources=[os.path.join(csrc_dir, 'qjl_gqa_score_kernel_xpu.cpp')],
            extra_compile_args={"cxx": sycl_flags()}
        ),
        SyclExtension(
            name='turboquant.xpu_quantization',
            sources=[os.path.join(csrc_dir, 'quantization_xpu.cpp')],
            extra_compile_args={"cxx": sycl_flags()}
        ),
    ]
    class XPUBuildExtension(BuildExtension):
        def build_extensions(self):
            if sys.platform == "win32":
                self.compiler.cc = 'icx'
                self.compiler.cxx = 'icx'
                original_spawn = self.compiler.spawn
                def spawn(cmd, **kwargs):
                    if cmd[0].endswith('cl.exe') or cmd[0].endswith('cl'):
                        cmd[0] = 'icx'
                        new_cmd = []
                        for arg in cmd:
                            if arg.startswith('-I') and ('Microsoft Visual Studio' in arg or 'Windows Kits' in arg):
                                new_cmd.append('-imsvc')
                                new_cmd.append(arg[2:])
                            elif arg.startswith('/I') and ('Microsoft Visual Studio' in arg or 'Windows Kits' in arg):
                                new_cmd.append('-imsvc')
                                new_cmd.append(arg[2:])
                            else:
                                new_cmd.append(arg)
                        cmd = new_cmd
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
