from setuptools import setup, find_packages

setup(
    name='CalTrig',
    version='0.1',
    author='CCI Ma Lab',
    author_email='michallange1995@gmail.com',
    description='CalTrig is a Toolbox for post-CNMF calcium imaging data analysis',
    url='https://github.com/cci-MaLab/Calcium-Transient-Analysis',
    packages=find_packages(),
    project_urls={
        "Documentation": "https://calcium-transient-analysis.readthedocs.io/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='3.10',
)