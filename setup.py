from setuptools import setup, find_packages

setup(
    name="kdezero",
    version='1.5',
    description='This library is an my improved version of "Deep Learning from Basic 3".',
    author='Kota Suzuki',
    author_email='suzuki.kota0331@gmail.com',
    url='https://github.com/kotabrog/K_DeZero.git',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
)
