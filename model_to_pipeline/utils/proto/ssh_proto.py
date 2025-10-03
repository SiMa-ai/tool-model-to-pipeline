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

from typing import Tuple
import paramiko
from scp import SCPClient


class SSH:
    def __init__(
        self,
        host: str,
        username: str = "root",
        password: str = "commitanddeliver",
        port: int = 22,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.initialize()

    def initialize(self) -> None:
        self.device = paramiko.SSHClient()
        self.device.set_missing_host_key_policy(policy=paramiko.AutoAddPolicy())
        self.device.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )
        self.scp = SCPClient(transport=self.device.get_transport(), socket_timeout=60)

    def execute_and_get_op(self, command: str) -> Tuple[str, int]:
        """Executes the command and returns the output and exitcode

        Args:
            command (str): Command to execute

        Returns:
            Tuple[str, int]: Output and exitcode
        """
        _, stdout, stderr = self.device.exec_command(command=command, timeout=90)
        output = stdout.readlines() + stderr.readlines()
        exit_code = stdout.channel.recv_exit_status()
        return "".join(output), exit_code

    def push(self, source: str, dest: str) -> None:
        """Pushes the source to dest in the remote

        Args:
            source (str): Source to push
            dest (str): Destination to push at
        """
        self.execute_and_get_op(command=f"touch {dest}; chown $USER {source}")
        self.scp.put(files=source, remote_path=dest, recursive=True)

    def exists(self, path: str) -> bool:
        return self.execute_and_get_op(command=f"ls {path}")[1] == 0
