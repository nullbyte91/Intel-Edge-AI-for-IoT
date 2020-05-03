## openvino optimization techniques:

### async with Multithread and multiprocess:

### Benchmark:
#### Hardware Configuration : i7-6820HQ CPU

|  Implementation |   FPS   |
|-----------------|---------|
| main_sync.py    | 9.20    |
| main_async.py   | 13.34   |
| main_async_multithread.py | 78.96|
