# Copyright (c) 2025 SiMa.ai
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from pathlib import Path
from setuptools import setup, find_packages


def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        print(f"Reading requirements from {requirements_path}")
        return requirements_path.read_text().splitlines()
    print("requirements.txt not found, using empty requirements.")
    return []


setup(
    description="A tool to convert SiMa.ai models to pipeline format",
    long_description=(
        "This package provides a command-line tool to convert SiMa.ai models into "
        "a pipeline format suitable for deployment on SiMa.ai hardware. It includes "
        "various surgeons and compilers to modify and optimize the model."
    ),
    name="sima-model-to-pipeline",
    version="0.1.0",
    author="SiMa.ai",
    author_email="siddhesh.sathe@sima.ai",
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            # "sima-model-to-pipeline=sima_model_to_pipeline.main:main",
            # "sima-fps-getter=sima_model.main:main",
            "sima-model-to-pipeline=sima_tool.main:main"
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
