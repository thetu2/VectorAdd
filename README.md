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

Using a single 32 bit float per element, the average time to add two 50000 element vectors, including transfer times to and from the gpu, is 0.198ms. The average time to add two complex vectors where each element is composed of two 32 bit floats is 0.285ms. Finally, the time to add vectors where each element is a single half2 type and also vectors where each element is two half types is 0.187ms and 0.191ms, respectively. 

When comparing the time between the real-only and the 32 bit complex vector additions, it surprising that the time is not doubled in the complex case given that there are twice as much data that is moved and twice as many calculations. This could mean that there are addition processes or overhead that require execution time on top of time for data transfers and arithmetic operations. 

When comparing the time between using two 32 bit floats and two 16 bit floats (2x half case), we notice a reduction in average time per trial as expected. Since this program was run on a Pascal based gpu, the gpu threads will treat the 16 bit floats the same way as 32 bit floats in terms of arithmetic operations. Therefore, this decrease in time per trial must have come from less data needing to be transferred between the gpu and the host. 

Theoretically, the gpu should be able to calculate results from the half2 vector additions twice as fast compared to using two separate half types. However, as only see a minor decrease in running time when using the half2 type. This result, in addition to the running times for the other data types, indicate that for this program, **data transfers to and from the gpu take up the majority of the time used**.
