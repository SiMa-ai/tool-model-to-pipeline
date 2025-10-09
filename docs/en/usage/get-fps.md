### get-fps

```sh
$ sima-model-to-pipeline get-fps --help

 Usage: sima-model-to-pipeline get-fps [OPTIONS]

 Finds the FPS number for MLA only using mla-rt

╭─ Options ──────────────────────────────────────────────────────────────╮
│ --device [davinci/modalix]   Type of the board to use for compilation. │
│                               [default: davinci]                       │
│                                                                        │
│ * --device-ip TEXT            Provide device IP address.               │
│                               [default: None] [required]               │
│                                                                        │
│ * --models TEXT               List of model paths (space-separated).   │
│                               [default: None] [required]               │
│                                                                        │
│ --help                        Show this message and exit.              │
╰─────────────────────────────────────────────────────────────────────-──╯

```