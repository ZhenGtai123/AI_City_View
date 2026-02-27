#!/usr/bin/env python3
"""
local_bench.py - 纯本地处理速度测试 (无上传)

1. 从 Azure 下载 N 张图到本地
2. 批量处理 (GPU + CPU)
3. 统计每张耗时、GPU利用率
"""
from __future__ import annotations

import copy
import sys
import os
import time
import shutil
import argparse
import queue
from pathlib import Path
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from main import process_panorama, get_default_config
from pipeline.stage2_ai_inference import set_gpu_concurrency, preload_models


def main():
    parser = argparse.ArgumentParser(description='本地处理速度测试')
    parser.add_argument('--azure-conn-str', default=os.environ.get('AZURE_STORAGE_CONNECTION_STRING', ''))
    parser.add_argument('--azure-container', default='')
    parser.add_argument('--azure-sas-url', default='')
    parser.add_argument('--azure-prefix', default='')
    parser.add_argument('-n', type=int, default=100, help='下载并处理的图片数')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu-concurrency', type=int, default=2)
    parser.add_argument('--io-threads', type=int, default=16)
    parser.add_argument('--buf', default='/tmp/ai_city_bench', help='本地缓冲目录')
    parser.add_argument('--skip-download', action='store_true', help='跳过下载，直接处理已有文件')
    args = parser.parse_args()

    buf = Path(args.buf)
    input_dir = buf / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)
    for wid in range(args.workers):
        (buf / f'w{wid}' / 'output').mkdir(parents=True, exist_ok=True)

    # ── Step 1: 下载 ──
    if not args.skip_download:
        from cloud_batch_run import _make_azure_client, list_azure_images, download_blob

        print(f"连接 Azure...", file=sys.stderr)
        client = _make_azure_client(args.azure_conn_str, args.azure_sas_url, args.azure_container)
        all_imgs = list_azure_images(client, args.azure_prefix)
        to_dl = all_imgs[:args.n]

        print(f"下载 {len(to_dl)} 张图片 ({args.io_threads} threads)...", file=sys.stderr)
        t0 = time.time()

        def _dl(blob_name):
            local = str(input_dir / os.path.basename(blob_name))
            download_blob(client, blob_name, local)
            return local

        with ThreadPoolExecutor(max_workers=args.io_threads) as pool:
            list(tqdm(pool.map(_dl, to_dl), total=len(to_dl), desc="Download", file=sys.stderr))

        dl_time = time.time() - t0
        print(f"下载完成: {len(to_dl)} 张, {dl_time:.1f}s ({dl_time/len(to_dl):.2f}s/张)", file=sys.stderr)
    else:
        print("跳过下载，使用已有文件", file=sys.stderr)

    # 列出本地文件
    local_files = sorted(input_dir.glob('*'))
    local_files = [f for f in local_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not local_files:
        print("没有找到图片文件!", file=sys.stderr)
        return
    print(f"待处理: {len(local_files)} 张", file=sys.stderr)

    # ── Step 2: 预加载模型 ──
    config = get_default_config()
    set_gpu_concurrency(args.gpu_concurrency)
    print(f"预加载模型 (workers={args.workers}, gpu_concurrency={args.gpu_concurrency})...", file=sys.stderr)
    preload_models(config)

    # ── Step 3: 批量处理 ──
    free_wids: queue.Queue[int] = queue.Queue()
    for w in range(args.workers):
        free_wids.put(w)

    ok_count = 0
    fail_count = 0
    times = []

    def _process(fpath: Path) -> Tuple[str, bool, str, float]:
        wid = free_wids.get()
        w_output = buf / f'w{wid}' / 'output'
        basename = fpath.stem
        t0 = time.time()
        try:
            cfg = copy.deepcopy(config)
            result = process_panorama(str(fpath), str(w_output), cfg)
            elapsed = time.time() - t0
            if not result.get('success'):
                return basename, False, result.get('error', 'unknown'), elapsed
            return basename, True, '', elapsed
        except Exception as e:
            return basename, False, str(e), time.time() - t0
        finally:
            free_wids.put(wid)
            # 清理输出
            for view in ('left', 'front', 'right'):
                vd = w_output / f"{basename}_{view}"
                if vd.exists():
                    shutil.rmtree(vd, ignore_errors=True)

    # 静默 stdout
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    print(f"\n开始处理...", file=sys.stderr)
    t_start = time.time()
    pbar = tqdm(total=len(local_files), desc="Process", unit="img", file=sys.stderr)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process, f): f.name for f in local_files}
            for f in as_completed(futures):
                basename, ok, msg, elapsed = f.result()
                times.append(elapsed)
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    f"ok={ok_count} fail={fail_count} last={elapsed:.1f}s"
                )
    finally:
        pbar.close()
        sys.stdout.close()
        sys.stdout = old_stdout

    # ── 统计 ──
    total_time = time.time() - t_start
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"处理完成: {ok_count} 成功, {fail_count} 失败", file=sys.stderr)
    print(f"总耗时: {total_time:.1f}s", file=sys.stderr)
    if times:
        import statistics
        print(f"每张: avg={statistics.mean(times):.1f}s, "
              f"median={statistics.median(times):.1f}s, "
              f"min={min(times):.1f}s, max={max(times):.1f}s", file=sys.stderr)
    if ok_count > 0:
        throughput = ok_count / total_time
        print(f"吞吐: {throughput:.2f} 张/s = {throughput*3600:.0f} 张/hr", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)


if __name__ == '__main__':
    main()
