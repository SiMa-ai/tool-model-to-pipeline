## Logs

The logs of execution will be dumped under directory named `logs/`. Every step involved in the tool execution, a separate log file will be generated based on the step name.

<br>

Example logs are as follows.

```sh
logs/
├── compile.log
├── main.log
├── pipelinecreate.log
└── surgery.log
```

The monitor server provides access to logs, which are updated in real time on the monitor UI, similar to the output of the `tail -f` command.