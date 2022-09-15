from setuptools import setup

setup(
    name='exoMMR',
    version='1.1.1',    
    description='A package focused on confirming and characterizing mean motion resonance in exoplanetary systems',
    url='https://github.com/shuds13/pyexample',
    author='Mariah MacDonald',
    author_email='mariah.g.macdonald@gmail.com',
    license='GPL',
    packages=['exoMMR'],
    install_requires=['matplotlib',
                      'numpy',
                      'rebound',
                      'pandas',
                      'astropy',
                      'scipy',
                      'progressbar',
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
)
