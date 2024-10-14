"""
Benchmark the latency of a given model. It accepts arguments similar to those of launch_server.py.

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --result-filename out.jsonl
## do some changes, and store the results under a different run_name:
python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --result-filename out.jsonl --run-name after
## plot the results in series of lines:
python -m sglang.bench_latency --result-filename out.jsonl --graph-sql="select run_name, batch_size, prefill_throughput from results"

# Usage (correctness test):
python -m sglang.bench_latency --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import dataclasses
import itertools
import logging
import multiprocessing
import os
import sqlite3
import time
from typing import Tuple
import nvtx

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import suppress_other_loggers


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "before"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = ""
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    # Plotting args
    graph_sql: str = (
        "select run_name, batch_size, prefill_throughput from results where run_name='before'"
    )
    graph_filename: str = "out.png"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        # graphing
        parser.add_argument("--graph-sql", type=str, default=BenchArgs.graph_sql)
        parser.add_argument(
            "--graph-filename", type=str, default=BenchArgs.graph_filename
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to case the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_config = ModelConfig(
        server_args.model_path,
        server_args.trust_remote_code,
        context_length=server_args.context_length,
    )
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        nccl_port=28888,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    input_ids = np.ones((batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(rid=i, origin_input_text="", origin_input_ids=list(input_ids[i]))
        req.prefix_indices = []
        req.sampling_params = sampling_params
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return reqs


@nvtx.annotate(message="extend", color="blue")
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        tree_cache=None,
    )
    batch.prepare_for_extend(model_runner.model_config.vocab_size)
    logits_output = model_runner.forward(batch)
    next_token_ids = model_runner.sample(logits_output, batch).tolist()
    return next_token_ids, logits_output.next_token_logits, batch


@nvtx.annotate(message="decode", color="orange")
def decode(input_token_ids, batch, model_runner):
    batch.prepare_for_decode(input_token_ids)
    logits_output = model_runner.forward(batch)
    next_token_ids = model_runner.sample(logits_output, batch).tolist()
    return next_token_ids, logits_output.next_token_logits

@nvtx.annotate(message="copy_data", color="red")
def copy_data(src_data, tar_data):
    # tar_data.copy_(src_data, non_blocking=True)

    with torch.profiler.record_function("copy_data"):
        # tar_data.copy_(src_data, non_blocking=True)
        tar_data.copy_(src_data)

    # with ThreadPoolExecutor() as executor:
    #     copy = executor.submit(
    #         tar_data.copy_, src_data
    #     )

@torch.inference_mode()
def overlap_test_run_once(
    run_name, model_runner, rank_print, reqs, batch_size, input_len, output_len,
    data, data_pin, data_gpu, data2
):
    # Clear the pools.
    # model_runner.req_to_token_pool.clear()
    # model_runner.token_to_kv_pool.clear()

    print("t0, data_pint[0,:10]: ", data_pin[0,:10])
    tick = time.time()
    copy_data(data, data_pin)
    print("t1, data_pint[0,:10]: ", data_pin[0,:10])
    print(f"Copy time: {time.time() - tick}")

    tick = time.time()
    data_pin[:] = 0
    print(f"clear data_pin time: {time.time() - tick}")
    print("t4, data_pint[0,:10]: ", data_pin[0,:10])

    tick = time.time()
    copy_data(data, data_pin)
    print("t3, data_pint[0,:10]: ", data_pin[0,:10])
    print(f"Copy time: {time.time() - tick}")

    # tick = time.time()
    # data_pin[:] = 0
    # print(f"clear data_pin time: {time.time() - tick}")
    # print("t2, data_pint[0,:10]: ", data_pin[0,:10])

    # tick = time.time()
    # copy_data(data2, data_pin)
    # print("t3, data_pint[0,:10]: ", data_pin[0,:10])
    # print(f"Copy time: {time.time() - tick}")

    # tick = time.time()
    # data_pin[:] = 0
    # print(f"clear data_pin time: {time.time() - tick}")
    # print("t4, data_pint[0,:10]: ", data_pin[0,:10])

    # tick = time.time()
    # copy_data(data2, data_pin)
    # print("t3, data_pint[0,:10]: ", data_pin[0,:10])
    # print(f"Copy time: {time.time() - tick}")

    tick = time.time()
    data_pin[:] = 0
    print(f"clear data_pin time: {time.time() - tick}")
    print("t4, data_pint[0,:10]: ", data_pin[0,:10])

    # for i in range(5):
    #     next_token_ids, _, batch = extend(reqs, model_runner)

    # tick = time.time()
    # next_token_ids, _, batch = extend(reqs, model_runner)
    # # Decode
    # for i in range(output_len):
    #     # if i == 1:
    #     #     with torch.cuda.stream(s1):
    #     #         data_gpu.copy_(data_pin, non_blocking=True)

    #     next_token_ids, _ = decode(next_token_ids, batch, model_runner)
    # print(f"LLM time 1: {time.time() - tick}")

    # tick = time.time()
    # next_token_ids, _, batch = extend(reqs, model_runner)
    # # Decode
    # for i in range(output_len):
    #     # if i == 1:
    #     #     with torch.cuda.stream(s1):
    #     #         data_gpu.copy_(data_pin, non_blocking=True)

    #     next_token_ids, _ = decode(next_token_ids, batch, model_runner)
    # print(f"LLM time 2: {time.time() - tick}")

    # s1 = torch.cuda.Stream()

    # with ThreadPoolExecutor() as executor:
    #     copy = executor.submit(
    #         copy_data, data, data_pin
    #     )
        # with torch.profiler.record_function("copy_data"):
        #     copy = executor.submit(
        #         copy_data, data, data_pin
        #     )

    # with torch.profiler.record_function("copy_data"):
    #     copy_data(data, data_pin)

    copy_thread = threading.Thread(target=copy_data, args=(data2, data_pin))
    tick = time.time()
    copy_thread.start()

    next_token_ids, _, batch = extend(reqs, model_runner)
    # Decode
    for i in range(output_len):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
    
    # copy.result()
    copy_thread.join()
    print(f"Copy + LLM time: {time.time() - tick}")
    print("t5, data_pint[0,:10]: ", data_pin[0,:10])

    return


def overlap_test(
    server_args,
    bench_args,
    tp_rank,
):
    print("Overlap Test!")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    n_data = 2*1024*1024
    h_dim = 768

    device = torch.device(f"cuda:{tp_rank}")
    # data = torch.ones(n_data, h_dim, dtype=torch.float16)
    data = torch.rand(n_data, h_dim, dtype=torch.float16)
    data2 = torch.rand(n_data, h_dim, dtype=torch.float16)
    data_cpu = torch.zeros(n_data, h_dim, dtype=torch.float16).pin_memory()
    data_gpu = torch.empty(n_data, h_dim, dtype=torch.float16, device=device)

    # Load the model
    model_runner, tokenizer = load_model(server_args, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0]
    )

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
             record_shapes=True, 
             profile_memory=True) as prof:
        # Warm up
        # rank_print("Warmup ...")
        overlap_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            bench_args.output_len[0],
            data,
            data_cpu,
            data_gpu,
            data2
        )
    # overlap_test_run_once(
    #     bench_args.run_name,
    #     model_runner,
    #     rank_print,
    #     reqs,
    #     bench_args.batch_size[0],
    #     bench_args.input_len[0],
    #     bench_args.output_len[0],
    #     data,
    #     data_cpu,
    #     data_gpu,
    # )

    torch.cuda.synchronize()
    # print(prof.key_averages().table())
    prof.export_chrome_trace("./overlap_trace/pytorch_trace_v2.6.json")
    # rank_print("Benchmark ...")
    print("Finished!!")


def main(server_args, bench_args):
    if server_args.model_path:
        work_func = overlap_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    if server_args.tp_size == 1:
        work_func(server_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    # For this script, model-path is not required
    assert (
        parser._actions[1].option_strings[0] == "--model-path"
    ), "options changed, this code need to be updated"
    parser._actions[1].required = False
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    main(server_args, bench_args)
