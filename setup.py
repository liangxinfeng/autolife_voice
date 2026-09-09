from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = 'autolife_s2_voice'


setup(
    name=PACKAGE_NAME,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{PACKAGE_NAME}']),
        (f'share/{PACKAGE_NAME}', ['package.xml', 'README.md']),
        (f'share/{PACKAGE_NAME}/launch', glob('launch/*.launch.py')),
    ],
    package_data={
        PACKAGE_NAME: [
            'configs/*.json',
            'configs/*.yaml',
            'configs/*.txt',
            'configs/https/*',
            'asset/*.pkl',
            'example/pkl/*.pkl',
        ],
    },
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Autolife',
    maintainer_email='ubuntu@example.com',
    description='Standalone Autolife S2 voice interaction chain.',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'voice = autolife_s2_voice.run.robot_voice:main',
            'voice_web = autolife_s2_voice.run.web_voice:main',
        ],
    },
)
