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

Make sure you selected `Model SDK`, `MPK CLI` and `Elxr SDK` during the SDK installation. To verify the SDK installation:

```bash
(.venv) sima-user@sima-user-machine:~/workspace/tool-model-to-pipeline$ sima-cli sdk ls
🔧 Environment: host (linux)
🖥️  Detected platform: Linux
✅ Docker daemon is running.

                 📦 Installed SDK Containers                 
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ SDK             ┃ Version                       ┃ Running ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ yocto           │ 2.0.0_Palette_SDK_master_B240 │   ✅    │
│ mpk_cli_toolset │ 2.0.0_Palette_SDK_master_B240 │   ✅    │
│ modelsdk        │ 2.0.0_Palette_SDK_master_B240 │   ✅    │
│ elxr            │ 2.0.0_Palette_SDK_master_B240 │   ✅    │
└─────────────────┴───────────────────────────────┴─────────┘
```

Then, install on the host machine in your workspace folder:

```bash
sima-user@sima-user-machine:~$ cd ~/workspace
sima-user@sima-user-machine:~$ sima-cli install gh:sima-ai/tool-model-to-pipeline
```

Change to your own local workspace folder if you are not using the default `~/workspace`.

This command will install the model-to-pipeline tool in three difference places:

1. The host machine itself
2. Model SDK container
3. MPK SDK container
