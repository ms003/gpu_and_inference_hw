import torch
from torch.profiler import ProfilerActivity, profile as torch_profile
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    """Generate tokens faster than the slow baseline.

    Main change
    - torch.inference_mode tells PyTorch we are not training, so it should
      not track gradients.
    - The first model call reads the whole prompt and saves the KV cache.
      After that, each step sends only the newest token plus the saved cache.
      This avoids recomputing the whole prompt again and again.
    - logits_to_keep=1 keeps only the last logits, because we only need the
      next-token prediction.
    - Tokens stay on the GPU during generation. We copy them to Python only
      once at the end. This avoids a slow .item() sync on every token.

    Expected impact: the KV cache is the biggest speedup. The other changes
    remove smaller per-token overheads and help keep the GPU busy.
    """

    if n_steps == 0:
        return []

    generated_tokens = torch.empty(
        n_steps, device=input_ids.device, dtype=input_ids.dtype
    )

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens[0] = next_token_id[0, 0]

        for step in range(1, n_steps):
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True
            )
            generated_tokens[step] = next_token_id[0, 0]

    return generated_tokens.tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    """Run a short profile and save a Chrome trace file.

    The printed table shows which PyTorch operations took the most time. The
    trace can be opened in Perfetto to see when the CPU launches GPU work and
    when the GPU is actually busy.
    """

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    torch.cuda.synchronize()
    with torch_profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
        torch.cuda.synchronize()

    sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=20))

    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace written to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    """Build, profile, and time the optimized version.

    The optimized model uses bfloat16. This uses less memory and is usually
    faster on modern NVIDIA GPUs like L40S and H100. The returned time is used
    to calculate the speedup versus the slow baseline.
    """

    torch.set_float32_matmul_precision("high")
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# 1. Used KV cache during generation.
#    The slow loop sends the whole growing sequence through the model every step.
#    The optimized loop runs the full prompt once, saves the KV cache, and then
#    sends only the newest token each step. This removes a lot of repeated work
#    and should be the main speedup.
#
# 2. Removed .item() inside the generation loop.
#    .item() copies one token from GPU to CPU every step and forces the CPU to
#    wait for the GPU. The optimized loop keeps tokens on the GPU and copies them
#    back once at the end.
#
# 3. Used logits_to_keep=1.
#    We only need the logits for the last token to choose the next token. This
#    avoids keeping logits for every prompt position.
#
# 4. Used bfloat16 for the optimized model.
#    bfloat16 uses less memory and is usually faster on L40S/H100 GPUs than
#    float32 for this kind of inference workload.
#

# Biggest impact and why:
#
# The biggest impact should come from using the KV cache. Without it, each new
# token recomputes attention over the whole prompt and all previously generated
# tokens. With the KV cache, the model reuses past key/value tensors and only
# processes one new token per step, so each decode step is much cheaper.
#