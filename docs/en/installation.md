# Installation

## Prerequisites

Before installing or running this tool, ensure that the following requirements are met:

- SiMa.ai DevKit connected to the network  
- A development PC running **Ubuntu 22.04**  
- [Palette SDK](https://docs.sima.ai/pages/palette/main.html) 2.0 properly installed  
- [Serial connection](https://docs.sima.ai/pages/overview/setup_standalone_mode/setup_serial.html) established prior to SDK usage for initial setup and debugging  

## Update `sima-cli`

Before installation, make sure your sima-cli is [up-to-date](https://docs.sima.ai/pages/sima_cli/main.html).

## Install the tool

First make sure you have installed the Palette SDK:
```bash
sima-user@sima-user-machine:~$ sima-cli install sdk
```

Then, install on the host machine in your workspace folder:

```bash
sima-user@sima-user-machine:~$ cd ~/workspace
sima-user@sima-user-machine:~$ sima-cli install gh:sima-ai/tool-model-to-pipeline
```

