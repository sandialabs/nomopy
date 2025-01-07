from setuptools import setup
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


with open("LICENSE.md", "r") as fh:
    license = fh.read()


setup(
      name='nomopy',
      version='0.1.0',
      description='Noise Modeling in Python',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='NA',
      author='modeling team',
      author_email='modelingteam@sandia.gov',
      license=license,
      classifiers=["Development Status :: 4 - Beta",
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS'],
      packages=find_packages(),
      setup_requires=['pytest-runner'],
      install_requires=['numpy >= 1.11.1',
                        'scipy >= 1.5.2',
                        'matplotlib >= 1.5.1',
                        'scikit-learn >= 0.23.2',
                        'joblib >= 0.17.0',
                        'autograd >= 1.3',
                       ],
      tests_require=['pytest',
                     'numpy',
                     'hypothesis'
                    ],
      python_requires='>=3.7',
      zip_safe=False,
)
