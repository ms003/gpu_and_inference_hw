import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # TODO (1 line): implement a lowest-AI op
    x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc=x
        for  _ in range(num_ops):
            acc =acc *x +x
        return acc

    # TODO (1 line): return either `fn` or `torch.compile(fn)` based on `compiled`
    return torch.compile (fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times=[]

    for  _  in range(rep):
        # create the two timing markers
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # we put the operation we want to time in between start/end
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        # with synchronize we are waitingfor any cpu work loading and GPU\
        # work to be completed so we only time the GPU time
        times.append(start.elapsed_time(end))
    return  torch.tensor(times).median().item()

    # TODO: time `rep` runs using CUDA events and return median latency (ms)
    # ============================================================================
    # Part 2: Benchmarking
    # ============================================================================
    #
    # TASK 2: Complete the benchmark function using CUDA events.
    # CUDA events measure GPU time precisely (not CPU wall time), which avoids
    # including kernel launch overhead or CPU-GPU synchronization delays.




# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # TODO: compute total FLOPs, arithmetic intensity, and achieved FLOP/s
    total_flops = num_elements * num_ops * 2

    if variant == 'compiled':
        total_bytes = num_elements * bytes_per_element * 2
    elif variant == 'eager':
        total_bytes = num_elements * bytes_per_element * 6 * num_ops
    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)

    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?

# Answer Q1:
# The performance rises because increasing numops makes the
# function repeat this operation more times for every element of
#     acc = acc * x + x. Each repeat does one multiply
#     and one add, so it adds about 2 more FLOPs per element.
#     In the compiled version, PyTorch can fuse this work
#     so the GPU mostly reads the input once, does all those repeated
#     calculations while the value is already inside the kernel, then writes
#     the final answer once. So the amount of math grows a lot, but the memory traffic
#     stays almost the same, which makes the measured FLOP/s rise.

# The measured runtime stays almost the same because the tensor size is the same
# for 1 ops and 64: the GPU reads the same input tensor and writes one output tensor
# of the same size. In the compiled version, it does not save every intermediate result
# back to memory. Instead, it keeps the temporary acc value inside the gpu kernel, does
# the repeated multiply/add operations there, and only writes the final result.
# So memory traffic stays almost the same, while the amount of math increases.
# Up to 64 , the extra math is still cheap enough that moving the data is the main cost.
#

# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.\
# Answer Q2:
# matmul 1024x1024 can be too small to fully use a big gpu like an H100.
# The gpu has many compute units, and this matrix multiply may not give
# it enough work to keep all of them busy. The 128  element-wise case
# uses a very large tensor, so the GPU has lots of independent elements
# to process in parallel. Because of that, the element-wise kernel can
# sometimes report higher flop/s than the smaller matrix multiply.

# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?

# Answer  Q3
#  in my implementation I did not really observe this as 64ops: 0.97ms and 128ops:0.87ms.
# However I was using 48GB or RAM. If I were to see the code gets slower would be because the GPU has
# to do a lot more calculations. The amount of data in memory is basically
# the same: it reads the same input tensor and writes the same output tensor.
# But for each value in the tensor, it repeats the multiply/add operation many more
# times. So the slowdown is not mainly from moving more data around; it is from
# doing more math. This means the GPU is starting to hit a memoery/RAM bottleneck.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
# Answer Q4
# The eager ops-K points look different because eager PyTorch runs the operations
# more separately. Each time you do acc = acc * x + x, eager mode may launch
# separate work for the multiply and the add, and it may write intermediate
# results back to memory. So it moves much more data and has more overhead.
# The compiled version can fuse those repeated operations into one optimized
# kernel, so it reads the input, does many calculations inside the kernel,
# and writes the final result. Because of that, the compiled version has much higher
# arithmetic intensity, while the eager version stays more memory-heavy.
