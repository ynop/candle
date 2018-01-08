from setuptools import setup

setup(name='candle',
      version='0.1',
      description='High level utility for training neural networks with pytorch.',
      url='',
      author='Matthias Buechi',
      author_email='buec@zhaw.ch',
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3 :: Only'
      ],
      keywords='',
      license='MIT',
      install_requires=[
          'numpy==1.14.0',
          'matplotlib==2.1.1'
      ],
      include_package_data=True,
      zip_safe=False,
      extras_require={
      },
      entry_points={
      }
      )
