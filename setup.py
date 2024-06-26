from glob import glob
from setuptools import setup
import os

package_name = 'lmpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, "launch"), glob('launch/*.launch.py')),
        # (os.path.join('share', package_name), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzangupenn',
    maintainer_email='zzang@seas.upenn.edu',
    description='f1tenth lmpc',
    license='MIT',
    tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': [
    #         'lmpc_node = lmpc.lmpc_node:main',
    #         'record_ss_node = lmpc.record_ss_node:main',
    #     ],
    # },
)
