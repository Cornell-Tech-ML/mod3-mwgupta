# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

* Parallel Check:
```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (163) 
---------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                          | 
        out: Storage,                                                                  | 
        out_shape: Shape,                                                              | 
        out_strides: Strides,                                                          | 
        in_storage: Storage,                                                           | 
        in_shape: Shape,                                                               | 
        in_strides: Strides,                                                           | 
    ) -> None:                                                                         | 
        # TODO: Implement for Task 3.1.                                                | 
        if np.array_equal(out_strides, in_strides) and len(in_storage) >= len(out):    | 
            for i in prange(len(out)):-------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                             | 
        else:                                                                          | 
            for i in prange(len(out)):-------------------------------------------------| #3
                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)------------------| #0
                in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)-------------------| #1
                to_index(i, out_shape, out_index)                                      | 
                broadcast_index(out_index, out_shape, in_shape, in_index)              | 
                o = index_to_position(out_index, out_strides)                          | 
                j = index_to_position(in_index, in_strides)                            | 
                out[o] = fn(in_storage[j])                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (177) is hoisted 
out of the parallel loop labelled #3 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (178) is hoisted 
out of the parallel loop labelled #3 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (211)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (211) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # TODO: Implement for Task 3.1.                                    | 
        if (                                                               | 
            np.array_equal(out_strides, a_strides)                         | 
            and len(a_storage) >= len(out)                                 | 
            and np.array_equal(a_strides, b_strides)                       | 
            and len(b_storage) >= len(out)                                 | 
        ):                                                                 | 
            for i in prange(len(out)):-------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        else:                                                              | 
            for i in prange(len(out)):-------------------------------------| #8
                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)------| #4
                a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #5
                b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #6
                to_index(i, out_shape, out_index)                          | 
                o = index_to_position(out_index, out_strides)              | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                j = index_to_position(a_index, a_strides)                  | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                k = index_to_position(b_index, b_strides)                  | 
                out[o] = fn(a_storage[j], b_storage[k])                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (233) is hoisted 
out of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (234) is hoisted 
out of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (235) is hoisted 
out of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (268)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (268) 
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     | 
        out: Storage,                                                | 
        out_shape: Shape,                                            | 
        out_strides: Strides,                                        | 
        a_storage: Storage,                                          | 
        a_shape: Shape,                                              | 
        a_strides: Strides,                                          | 
        reduce_dim: int,                                             | 
    ) -> None:                                                       | 
        # TODO: Implement for Task 3.1.                              | 
        for i in prange(len(out)):-----------------------------------| #10
            out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)----| #9
            reduce_size = a_shape[reduce_dim]                        | 
            to_index(i, out_shape, out_index)                        | 
            o = index_to_position(out_index, out_strides)            | 
            acc = out[o]                                             | 
            for s in range(reduce_size):                             | 
                out_index[reduce_dim] = s                            | 
                j = index_to_position(out_index, a_strides)          | 
                acc = fn(acc, a_storage[j])                          | 
            out[o] = acc                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (279) is hoisted 
out of the parallel loop labelled #10 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (293)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/meiligupta/workspace/mod3-mwgupta/minitorch/fast_ops.py (293) 
--------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                          | 
    out: Storage,                                                                     | 
    out_shape: Shape,                                                                 | 
    out_strides: Strides,                                                             | 
    a_storage: Storage,                                                               | 
    a_shape: Shape,                                                                   | 
    a_strides: Strides,                                                               | 
    b_storage: Storage,                                                               | 
    b_shape: Shape,                                                                   | 
    b_strides: Strides,                                                               | 
) -> None:                                                                            | 
    """NUMBA tensor matrix multiply function.                                         | 
                                                                                      | 
    Should work for any tensor shapes that broadcast as long as                       | 
                                                                                      | 
    ```                                                                               | 
    assert a_shape[-1] == b_shape[-2]                                                 | 
    ```                                                                               | 
                                                                                      | 
    Optimizations:                                                                    | 
                                                                                      | 
    * Outer loop in parallel                                                          | 
    * No index buffers or function calls                                              | 
    * Inner loop should have no global writes, 1 multiply.                            | 
                                                                                      | 
                                                                                      | 
    Args:                                                                             | 
    ----                                                                              | 
        out (Storage): storage for `out` tensor                                       | 
        out_shape (Shape): shape for `out` tensor                                     | 
        out_strides (Strides): strides for `out` tensor                               | 
        a_storage (Storage): storage for `a` tensor                                   | 
        a_shape (Shape): shape for `a` tensor                                         | 
        a_strides (Strides): strides for `a` tensor                                   | 
        b_storage (Storage): storage for `b` tensor                                   | 
        b_shape (Shape): shape for `b` tensor                                         | 
        b_strides (Strides): strides for `b` tensor                                   | 
                                                                                      | 
    Returns:                                                                          | 
    -------                                                                           | 
        None : Fills in `out`                                                         | 
                                                                                      | 
    """                                                                               | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                            | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                            | 
                                                                                      | 
    # TODO: Implement for Task 3.2.                                                   | 
    m = a_shape[-2]  # Rows of A                                                      | 
    n = b_shape[-1]  # Columns of B                                                   | 
    k = a_shape[-1]  # Shared dimension                                               | 
    for i in prange(len(out)):--------------------------------------------------------| #11
        batch = (i // (m * n)) if len(out_shape) > 2 else 0                           | 
        r = (i // n) % m                                                              | 
        c = i % n                                                                     | 
                                                                                      | 
        acc = 0.0                                                                     | 
        for j in range(k):                                                            | 
            a_pos = batch * a_batch_stride + r * a_strides[-2] + j * a_strides[-1]    | 
            b_pos = batch * b_batch_stride + j * b_strides[-2] + c * b_strides[-1]    | 
            acc += a_storage[a_pos] * b_storage[b_pos]                                | 
                                                                                      | 
        out[i] = acc                                                                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

* CPU, Simple

```
Epoch 0 | Loss: 5.8665 | Correct: 40 | Time: 6.80 seconds
Epoch 10 | Loss: 1.3974 | Correct: 49 | Time: 0.20 seconds
Epoch 20 | Loss: 1.7452 | Correct: 50 | Time: 0.19 seconds
Epoch 30 | Loss: 1.3280 | Correct: 49 | Time: 0.20 seconds
Epoch 40 | Loss: 0.9441 | Correct: 50 | Time: 0.27 seconds
Epoch 50 | Loss: 0.3486 | Correct: 50 | Time: 0.20 seconds
Epoch 60 | Loss: 0.2661 | Correct: 50 | Time: 0.19 seconds
Epoch 70 | Loss: 1.0671 | Correct: 50 | Time: 0.20 seconds
Epoch 80 | Loss: 0.6021 | Correct: 50 | Time: 0.20 seconds
Epoch 90 | Loss: 0.4646 | Correct: 50 | Time: 0.19 seconds
Epoch 100 | Loss: 0.0935 | Correct: 50 | Time: 0.21 seconds
Epoch 110 | Loss: 0.4653 | Correct: 50 | Time: 0.19 seconds
Epoch 120 | Loss: 0.0343 | Correct: 50 | Time: 0.19 seconds
Epoch 130 | Loss: 0.5706 | Correct: 50 | Time: 0.19 seconds
Epoch 140 | Loss: 0.9998 | Correct: 50 | Time: 0.19 seconds
Epoch 150 | Loss: 0.6199 | Correct: 50 | Time: 0.19 seconds
Epoch 160 | Loss: 0.5890 | Correct: 50 | Time: 0.19 seconds
Epoch 170 | Loss: 0.3500 | Correct: 50 | Time: 0.19 seconds
Epoch 180 | Loss: 0.2839 | Correct: 50 | Time: 0.19 seconds
Epoch 190 | Loss: 0.3119 | Correct: 50 | Time: 0.21 seconds
Epoch 200 | Loss: 0.1343 | Correct: 50 | Time: 0.19 seconds
Epoch 210 | Loss: 0.1726 | Correct: 50 | Time: 0.19 seconds
Epoch 220 | Loss: 0.0976 | Correct: 50 | Time: 0.19 seconds
Epoch 230 | Loss: 0.0261 | Correct: 50 | Time: 0.19 seconds
Epoch 240 | Loss: 0.6587 | Correct: 50 | Time: 0.33 seconds
Epoch 250 | Loss: 0.3990 | Correct: 50 | Time: 0.24 seconds
Epoch 260 | Loss: 0.6427 | Correct: 50 | Time: 0.19 seconds
Epoch 270 | Loss: 0.6527 | Correct: 50 | Time: 0.22 seconds
Epoch 280 | Loss: 0.0199 | Correct: 50 | Time: 0.20 seconds
Epoch 290 | Loss: 0.2231 | Correct: 50 | Time: 0.19 seconds
Epoch 300 | Loss: 0.0064 | Correct: 50 | Time: 0.21 seconds
Epoch 310 | Loss: 0.2417 | Correct: 50 | Time: 0.19 seconds
Epoch 320 | Loss: 0.4476 | Correct: 50 | Time: 0.22 seconds
Epoch 330 | Loss: 0.5823 | Correct: 50 | Time: 0.19 seconds
Epoch 340 | Loss: 0.3827 | Correct: 50 | Time: 0.20 seconds
Epoch 350 | Loss: 0.5919 | Correct: 50 | Time: 0.20 seconds
Epoch 360 | Loss: 0.1326 | Correct: 50 | Time: 0.27 seconds
Epoch 370 | Loss: 0.1300 | Correct: 50 | Time: 0.19 seconds
Epoch 380 | Loss: 0.4136 | Correct: 50 | Time: 0.21 seconds
Epoch 390 | Loss: 0.1609 | Correct: 50 | Time: 0.21 seconds
Epoch 400 | Loss: 0.3112 | Correct: 50 | Time: 0.21 seconds
Epoch 410 | Loss: 0.0597 | Correct: 50 | Time: 0.19 seconds
Epoch 420 | Loss: 0.0044 | Correct: 50 | Time: 0.21 seconds
Epoch 430 | Loss: 0.0003 | Correct: 50 | Time: 0.19 seconds
Epoch 440 | Loss: 0.1265 | Correct: 50 | Time: 0.19 seconds
Epoch 450 | Loss: 0.1544 | Correct: 50 | Time: 0.19 seconds
Epoch 460 | Loss: 0.0049 | Correct: 50 | Time: 0.19 seconds
Epoch 470 | Loss: 0.1640 | Correct: 50 | Time: 0.19 seconds
Epoch 480 | Loss: 0.1874 | Correct: 50 | Time: 0.19 seconds
Epoch 490 | Loss: 0.1442 | Correct: 50 | Time: 0.19 seconds
Epoch 499 | Loss: 0.1520 | Correct: 50 | Time: 0.20 seconds
```

* CPU, Split

```
Epoch 0 | Loss: 6.7201 | Correct: 28 | Time: 6.25 seconds
Epoch 10 | Loss: 5.9826 | Correct: 38 | Time: 0.19 seconds
Epoch 20 | Loss: 5.9277 | Correct: 36 | Time: 0.19 seconds
Epoch 30 | Loss: 3.9810 | Correct: 40 | Time: 0.19 seconds
Epoch 40 | Loss: 4.1988 | Correct: 40 | Time: 0.21 seconds
Epoch 50 | Loss: 4.0087 | Correct: 42 | Time: 0.19 seconds
Epoch 60 | Loss: 2.0116 | Correct: 46 | Time: 0.19 seconds
Epoch 70 | Loss: 1.3094 | Correct: 46 | Time: 0.20 seconds
Epoch 80 | Loss: 2.4334 | Correct: 50 | Time: 0.19 seconds
Epoch 90 | Loss: 1.9314 | Correct: 50 | Time: 0.19 seconds
Epoch 100 | Loss: 1.7225 | Correct: 50 | Time: 0.19 seconds
Epoch 110 | Loss: 2.1559 | Correct: 48 | Time: 0.19 seconds
Epoch 120 | Loss: 0.3092 | Correct: 50 | Time: 0.19 seconds
Epoch 130 | Loss: 1.3053 | Correct: 50 | Time: 0.19 seconds
Epoch 140 | Loss: 0.5218 | Correct: 50 | Time: 0.19 seconds
Epoch 150 | Loss: 0.9026 | Correct: 50 | Time: 0.19 seconds
Epoch 160 | Loss: 0.7833 | Correct: 50 | Time: 0.19 seconds
Epoch 170 | Loss: 0.4613 | Correct: 50 | Time: 0.19 seconds
Epoch 180 | Loss: 1.0722 | Correct: 50 | Time: 0.19 seconds
Epoch 190 | Loss: 1.3552 | Correct: 50 | Time: 0.19 seconds
Epoch 200 | Loss: 0.8875 | Correct: 50 | Time: 0.19 seconds
Epoch 210 | Loss: 1.2018 | Correct: 50 | Time: 0.20 seconds
Epoch 220 | Loss: 0.9889 | Correct: 50 | Time: 0.19 seconds
Epoch 230 | Loss: 0.1052 | Correct: 50 | Time: 0.19 seconds
Epoch 240 | Loss: 0.5823 | Correct: 50 | Time: 0.19 seconds
Epoch 250 | Loss: 0.7255 | Correct: 50 | Time: 0.19 seconds
Epoch 260 | Loss: 0.8228 | Correct: 50 | Time: 0.19 seconds
Epoch 270 | Loss: 0.5113 | Correct: 50 | Time: 0.19 seconds
Epoch 280 | Loss: 0.3385 | Correct: 50 | Time: 0.19 seconds
Epoch 290 | Loss: 0.8699 | Correct: 50 | Time: 0.19 seconds
Epoch 300 | Loss: 0.1081 | Correct: 50 | Time: 0.19 seconds
Epoch 310 | Loss: 0.3961 | Correct: 50 | Time: 0.19 seconds
Epoch 320 | Loss: 0.3291 | Correct: 50 | Time: 0.19 seconds
Epoch 330 | Loss: 0.1894 | Correct: 50 | Time: 0.19 seconds
Epoch 340 | Loss: 0.0580 | Correct: 50 | Time: 0.19 seconds
Epoch 350 | Loss: 0.1100 | Correct: 50 | Time: 0.19 seconds
Epoch 360 | Loss: 0.2196 | Correct: 50 | Time: 0.19 seconds
Epoch 370 | Loss: 0.5135 | Correct: 50 | Time: 0.19 seconds
Epoch 380 | Loss: 0.8901 | Correct: 50 | Time: 0.19 seconds
Epoch 390 | Loss: 0.0303 | Correct: 50 | Time: 0.21 seconds
Epoch 400 | Loss: 0.3385 | Correct: 50 | Time: 0.23 seconds
Epoch 410 | Loss: 0.4656 | Correct: 50 | Time: 0.20 seconds
Epoch 420 | Loss: 0.1439 | Correct: 50 | Time: 0.19 seconds
Epoch 430 | Loss: 0.1378 | Correct: 50 | Time: 0.20 seconds
Epoch 440 | Loss: 0.3694 | Correct: 50 | Time: 0.28 seconds
Epoch 450 | Loss: 0.5054 | Correct: 50 | Time: 0.19 seconds
Epoch 460 | Loss: 0.0979 | Correct: 50 | Time: 0.21 seconds
Epoch 470 | Loss: 0.0213 | Correct: 50 | Time: 0.19 seconds
Epoch 480 | Loss: 0.6708 | Correct: 50 | Time: 0.19 seconds
Epoch 490 | Loss: 0.3456 | Correct: 50 | Time: 0.19 seconds
Epoch 499 | Loss: 0.1588 | Correct: 50 | Time: 0.23 seconds
```

* CPU, Xor

```
Epoch 0 | Loss: 7.2767 | Correct: 28 | Time: 6.54 seconds
Epoch 10 | Loss: 5.2042 | Correct: 36 | Time: 0.19 seconds
Epoch 20 | Loss: 4.6998 | Correct: 40 | Time: 0.21 seconds
Epoch 30 | Loss: 5.3045 | Correct: 43 | Time: 0.20 seconds
Epoch 40 | Loss: 2.9087 | Correct: 46 | Time: 0.20 seconds
Epoch 50 | Loss: 4.2957 | Correct: 49 | Time: 0.20 seconds
Epoch 60 | Loss: 1.7794 | Correct: 50 | Time: 0.19 seconds
Epoch 70 | Loss: 1.2223 | Correct: 49 | Time: 0.19 seconds
Epoch 80 | Loss: 1.8534 | Correct: 50 | Time: 0.19 seconds
Epoch 90 | Loss: 2.1686 | Correct: 48 | Time: 0.20 seconds
Epoch 100 | Loss: 1.7364 | Correct: 50 | Time: 0.20 seconds
Epoch 110 | Loss: 1.0176 | Correct: 50 | Time: 0.19 seconds
Epoch 120 | Loss: 1.4763 | Correct: 50 | Time: 0.19 seconds
Epoch 130 | Loss: 1.6515 | Correct: 50 | Time: 0.22 seconds
Epoch 140 | Loss: 0.5706 | Correct: 50 | Time: 0.19 seconds
Epoch 150 | Loss: 0.6191 | Correct: 50 | Time: 0.19 seconds
Epoch 160 | Loss: 0.9487 | Correct: 50 | Time: 0.19 seconds
Epoch 170 | Loss: 0.9691 | Correct: 50 | Time: 0.20 seconds
Epoch 180 | Loss: 1.1773 | Correct: 50 | Time: 0.20 seconds
Epoch 190 | Loss: 0.5833 | Correct: 50 | Time: 0.20 seconds
Epoch 200 | Loss: 0.3337 | Correct: 50 | Time: 0.19 seconds
Epoch 210 | Loss: 0.9669 | Correct: 50 | Time: 0.19 seconds
Epoch 220 | Loss: 0.8130 | Correct: 50 | Time: 0.21 seconds
Epoch 230 | Loss: 0.4438 | Correct: 50 | Time: 0.19 seconds
Epoch 240 | Loss: 1.1705 | Correct: 50 | Time: 0.19 seconds
Epoch 250 | Loss: 0.5125 | Correct: 50 | Time: 0.19 seconds
Epoch 260 | Loss: 0.7450 | Correct: 50 | Time: 0.21 seconds
Epoch 270 | Loss: 0.2498 | Correct: 50 | Time: 0.20 seconds
Epoch 280 | Loss: 0.4743 | Correct: 50 | Time: 0.20 seconds
Epoch 290 | Loss: 0.7085 | Correct: 50 | Time: 0.19 seconds
Epoch 300 | Loss: 0.1698 | Correct: 50 | Time: 0.19 seconds
Epoch 310 | Loss: 0.1291 | Correct: 50 | Time: 0.19 seconds
Epoch 320 | Loss: 0.4564 | Correct: 50 | Time: 0.19 seconds
Epoch 330 | Loss: 0.2578 | Correct: 50 | Time: 0.19 seconds
Epoch 340 | Loss: 0.1836 | Correct: 50 | Time: 0.19 seconds
Epoch 350 | Loss: 0.3410 | Correct: 50 | Time: 0.19 seconds
Epoch 360 | Loss: 0.2207 | Correct: 50 | Time: 0.19 seconds
Epoch 370 | Loss: 0.4235 | Correct: 50 | Time: 0.19 seconds
Epoch 380 | Loss: 0.1232 | Correct: 50 | Time: 0.19 seconds
Epoch 390 | Loss: 0.0683 | Correct: 50 | Time: 0.19 seconds
Epoch 400 | Loss: 0.2808 | Correct: 50 | Time: 0.19 seconds
Epoch 410 | Loss: 0.1827 | Correct: 50 | Time: 0.20 seconds
Epoch 420 | Loss: 0.1488 | Correct: 50 | Time: 0.26 seconds
Epoch 430 | Loss: 0.0375 | Correct: 50 | Time: 0.19 seconds
Epoch 440 | Loss: 0.0439 | Correct: 50 | Time: 0.20 seconds
Epoch 450 | Loss: 0.3431 | Correct: 50 | Time: 0.19 seconds
Epoch 460 | Loss: 0.2017 | Correct: 50 | Time: 0.19 seconds
Epoch 470 | Loss: 0.2375 | Correct: 50 | Time: 0.19 seconds
Epoch 480 | Loss: 0.1813 | Correct: 50 | Time: 0.19 seconds
Epoch 490 | Loss: 0.0951 | Correct: 50 | Time: 0.21 seconds
Epoch 499 | Loss: 0.2848 | Correct: 50 | Time: 0.19 seconds
```

* GPU, Simple

```
```

* GPU, Split

```
```

* GPU, XOR

```
```