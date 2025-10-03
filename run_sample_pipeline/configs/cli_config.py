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


from pydantic import BaseModel, Field


class Config(BaseModel):
    """Config for infer command"""

    device: str = Field("davinci", description="Device type")
    device_ip: str = Field(..., description="Provide device IP address")
    model: str = Field(..., description="Model path to create and run pipeline on")
    rtsp_src: str = Field(..., description="RTSP Stream URL")
    host_ip: str = Field(..., description="IP address of the host to visualize")
    host_port: int = Field(7000, description="Port to visualize the pipeline on")
    labels_file: str = Field(
        None,
        description="Path to labels file. If not provided, default YOLO 80 classes will be used",
    )
    pipeline_name: str = Field("MyPipeline", description="Name the pipeline")
