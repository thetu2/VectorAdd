# VectorAdd
## Build Instructions (Visual Studio 2017+)

- Open folder containing project files using VS
- Build->Build All
- Select startup item->vector_add_perf.exe


## Operating Instuctions
The point of this code is to test various datatypes including a half precision complex type. To select the datatype to test, set the `TYPE` macro on line 26.

## Results
The code was run using an NVIDIA 1080 GPU. The number of trials was 10000 and the number of elements to add per trial was 50000.


### Real
```
[Vector addition of 50000 elements]
CUDA kernel launch with 196 blocks of 256 threads
Sample output on index 6: 0.710501+0.513535=1.224036
Total elapsed time including kernel execution and mem transfers from device to host: 1981.716064 ms
Average time per trial: 0.198172 ms
Done

C:\Users\tarve\Desktop\VectorAdd\out\build\x64-Debug\vector_add_perf.exe (process 186352) exited with code 0.
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .

```

### complex32
```
[Vector addition of 50000 elements]
CUDA kernel launch with 196 blocks of 256 threads
Sample output on index 6: (0.008911+0.377880i)+(0.531663+0.571184i)=0.540574+0.949065i
Total elapsed time including kernel execution and mem transfers from device to host: 2846.096191 ms
Average time per trial: 0.284610 ms
Done

C:\Users\tarve\Desktop\VectorAdd\out\build\x64-Debug\vector_add_perf.exe (process 179604) exited with code 0.
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .

```

### half2
```
[Vector addition of 50000 elements]
CUDA kernel launch with 196 blocks of 256 threads
Sample output on index 6: (0.008911+0.377930i)+(0.531738+0.571289i)=0.540527+0.949219i
Total elapsed time including kernel execution and mem transfers from device to host: 1868.072144 ms
Average time per trial: 0.186807 ms
Done

C:\Users\tarve\Desktop\VectorAdd\out\build\x64-Debug\vector_add_perf.exe (process 176620) exited with code 0.
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .
```

### 2x half
```
[Vector addition of 50000 elements]
CUDA kernel launch with 196 blocks of 256 threads
Sample output on index 6: (0.008911+0.377930i)+(0.531738+0.571289i)=0.540527+0.949219i
Total elapsed time including kernel execution and mem transfers from device to host: 1909.880981 ms
Average time per trial: 0.190988 ms
Done

C:\Users\tarve\Desktop\VectorAdd\out\build\x64-Debug\vector_add_perf.exe (process 185400) exited with code 0.
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .
```

## Conclusions.

For Jerry to fill out