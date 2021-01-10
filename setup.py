#from distutils.core import setup, Extension

from numpy.distutils.core import setup, Extension
import numpy
import os

rootpath = os.path.abspath('./')

os.chdir('./beam_cxx')
os.makedirs('build', exist_ok=True)
os.chdir('build')
os.system('cmake ../ -DCMAKE_INSTALL_PREFIX=..')
os.system('make install')

os.chdir(rootpath)

print(rootpath)
library_dirs = [os.path.join(rootpath, 'beam_cxx/lib')]
module1 = Extension(
    '_simtod',
    language='c++',
    sources = ['./src/_tod.cxx'],
    include_dirs=[numpy.get_include(), os.path.join(rootpath, 'beam_cxx/include')],
    library_dirs=library_dirs,
    libraries=['convolve'],
    extra_compile_args=['-std=c++11', '-march=native'],
    extra_link_args=['-Wl,-rpath,' + i for i in library_dirs]
)

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])
