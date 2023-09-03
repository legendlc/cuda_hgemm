# hgemm performance

## naive SIMT

```shell
[HGEMM 2023-09-03 17:59:35 E:\Code\CUDA\cuda_hgemm\src\include\matrix.h:38 Matrix::Matrix] Matrix A: 4096 * 4096, cpu: 000001843E814040, gpu: 0000000505600000
[HGEMM 2023-09-03 17:59:36 E:\Code\CUDA\cuda_hgemm\src\include\matrix.h:38 Matrix::Matrix] Matrix B: 4096 * 4096, cpu: 0000018442823040, gpu: 0000000507600000
[HGEMM 2023-09-03 17:59:37 E:\Code\CUDA\cuda_hgemm\src\include\matrix.h:38 Matrix::Matrix] Matrix C: 4096 * 4096, cpu: 000001844683F040, gpu: 0000000509600000
[HGEMM 2023-09-03 17:59:37 E:\Code\CUDA\cuda_hgemm\src\include\matrix.h:38 Matrix::Matrix] Matrix Base: 4096 * 4096, cpu: 000001844A85A040, gpu: 000000050B600000
[HGEMM 2023-09-03 17:59:37 include\tester.h:75 Tester::evaluate] ----------------- Evaluating Simt-Naive -----------------
[HGEMM 2023-09-03 17:59:39 include\tester.h:85 Tester::evaluate] Warm up time: 1193.959 ms
[HGEMM 2023-09-03 17:59:50 include\tester.h:126 Tester::profile] Simt-Naive exit, profiling time: 1166.960 ms (100.00%), throughput: 0.118 TFLOPS (100.00%)
```
