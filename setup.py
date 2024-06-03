import os
import shutil
from setuptools import setup, Extension
from setuptools.command.install import install

class InstallCMakeModule(install):
    def run(self):
        # Define the build directory and the output module name
        build_dir = 'build'
        module_name = 'pyGpuBM3D'
        module_filename = 'pyGpuBM3D.cpython-310-x86_64-linux-gnu.so'  # Adjust based on your module's actual filename

        # Run the standard install process
        install.run(self)

        # Copy the built module to the appropriate location in the installation directory
        target_dir = os.path.join(self.install_lib, module_name)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(build_dir, module_filename),
            os.path.join(target_dir, module_filename)
        )

setup(
    name='pyGpuBM3D',
    version='0.1',
    description='GPU implementation of BM3D denoising.',
    packages=['pyGpuBM3D'],
    package_dir={'': 'python'},
    cmdclass={'install': InstallCMakeModule},
)