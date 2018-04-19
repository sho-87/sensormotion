from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(name='sensormotion',
      version='1.0.1',
      description='Python package for analyzing sensor-collected human motion '
                  'data (e.g. physical activity levels, gait dynamics)',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/sho-87/sensormotion',
      project_urls={
          'Documentation': 'http://sensormotion.readthedocs.io',
          'Source': 'https://github.com/sho-87/sensormotion',
          'Tracker': 'https://github.com/sho-87/sensormotion/issues',
      },
      author='Simon Ho',
      author_email='simonho213@gmail.com',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3',
      install_requires=['matplotlib',
                        'numpy',
                        'scipy'],
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
      ],
      keywords='gait accelerometer signal-processing walking actigraph'
               ' physical-activity',
      zip_safe=True)
