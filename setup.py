from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kdezero",
    version='2.1',
    description='This library is an my improved version of "Deep Learning from Basic 3".',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author='Kota Suzuki',
    author_email='suzuki.kota0331@gmail.com',
    url='https://github.com/kotabrog/K_DeZero.git',
    license='MIT',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
)
