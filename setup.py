from setuptools import setup, find_packages

setup(
    name='chatfusion',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
    ],
    author='Qusai Albonni',
    author_email='albonniqusai@gmail.com',
    description='A flexible and powerful Python library for interacting with various AI language models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/QusaiAlbonni/chatfusion',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    python_requires='>=3.7',
)
