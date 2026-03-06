
from setuptools import setup, find_packages

package_name = 'lidar_gp_cbf'
submodules = 'lidar_gp_cbf/control_lib'
scenarios = 'lidar_gp_cbf/scenarios'
simulator = 'lidar_gp_cbf/simulator'

setup(
    name=package_name,
    version='0.0.1',
    #packages=find_packages(include=[package_name, f'{package_name}.*']),
    packages=[package_name, submodules, scenarios, simulator],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shaghayegh Keyumarsi',
    maintainer_email='s.keyumarsi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'expROS2_main = lidar_gp_cbf.expROS2_main:main'
        ],
    },
)
