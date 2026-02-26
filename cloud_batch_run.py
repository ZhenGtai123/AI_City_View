#!/usr/bin/env python3
"""
cloud_batch_run.py - Azure Blob → GCP VM → GCS

处理流程:
1. 从 Azure Blob Storage 列出所有全景图
2. 检查 GCS 输出桶，跳过已完成的图片（断点续跑）
3. 逐张下载到本地 SSD 缓冲区
4. 运行完整 pipeline (3视角 × 23张输出)
5. 上传结果到 GCS
6. 清理本地文件
7. 捕获 SIGTERM 实现 Spot 抢占优雅退出

用法:
  python cloud_batch_run.py \
    --azure-conn-str "DefaultEndpointsProtocol=..." \
    --azure-container panoramas \
    --gcs-bucket my-ai-city-output \
    --gcs-prefix output \
    --limit 0

  # 或使用 SAS URL
  python cloud_batch_run.py \
    --azure-sas-url "https://account.blob.core.windows.net/container?sv=..." \
    --gcs-bucket my-ai-city-output

环境变量:
  AZURE_STORAGE_CONNECTION_STRING  Azure 连接字符串
  GOOGLE_APPLICATION_CREDENTIALS   GCP 服务账号密钥 (VM 上不需要)
"""

from __future__ import annotations

import contextlib
import copy
import sys
import os
import signal
import time
import shutil
import argparse
import logging
from pathlib import Path
from typing import Set, List, Optional, Tuple
from threading import Event
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED
from concurrent.futures import wait as futures_wait

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from main import process_panorama, get_default_config
from pipeline.stage2_ai_inference import set_gpu_concurrency

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
CHECKPOINT_INTERVAL = 10  # 每处理 N 张打印一次统计
UPLOAD_MAX_RETRIES = 3
UPLOAD_RETRY_DELAY = 2  # seconds
AZURE_DOWNLOAD_TIMEOUT = 120  # seconds

# ---------------------------------------------------------------------------
# Graceful shutdown on Spot preemption (SIGTERM) or Ctrl-C (SIGINT)
# ---------------------------------------------------------------------------
_shutdown_event = Event()


def _signal_handler(signum, frame):
    _shutdown_event.set()
    logging.warning("收到信号 %s，等待当前图片完成后安全退出...", signum)


# ---------------------------------------------------------------------------
# Azure Blob helpers
# ---------------------------------------------------------------------------
def _make_azure_client(conn_str: Optional[str], sas_url: Optional[str], container: Optional[str]):
    """创建 Azure ContainerClient (支持 connection string 或 SAS URL)"""
    from azure.storage.blob import ContainerClient

    if sas_url:
        return ContainerClient.from_container_url(sas_url)
    if conn_str and container:
        return ContainerClient.from_connection_string(conn_str, container)
    raise ValueError("需要 --azure-conn-str + --azure-container 或 --azure-sas-url")


BLOB_LIST_CACHE = Path(__file__).parent / '.blob_list_cache.txt'


def list_azure_images(client, prefix: str = '', force_refresh: bool = False) -> List[str]:
    """列出 Azure 容器中所有图片 blob 名称 (带本地缓存)"""
    # 有缓存直接读
    if not force_refresh and BLOB_LIST_CACHE.exists():
        blobs = [l.strip() for l in BLOB_LIST_CACHE.read_text().splitlines() if l.strip()]
        logging.info("从缓存加载 %d 张图片 (%s)", len(blobs), BLOB_LIST_CACHE.name)
        return blobs

    # 首次: 从 Azure 列出并缓存
    logging.info("首次列出 Azure Blob (可能需要几分钟)...")
    blobs = []
    count = 0
    for blob in client.list_blobs(name_starts_with=prefix or None):
        if Path(blob.name).suffix.lower() in IMAGE_EXTENSIONS:
            blobs.append(blob.name)
        count += 1
        if count % 50000 == 0:
            logging.info("  已扫描 %d 个 blob, 找到 %d 张图片...", count, len(blobs))
    blobs.sort()

    # 写入缓存
    BLOB_LIST_CACHE.write_text('\n'.join(blobs))
    logging.info("Azure Blob 共找到 %d 张图片 (已缓存到 %s)", len(blobs), BLOB_LIST_CACHE.name)
    return blobs


def download_blob(client, blob_name: str, local_path: str) -> None:
    """下载单个 blob 到本地文件 (带超时)"""
    with open(local_path, 'wb') as f:
        data = client.download_blob(blob_name, timeout=AZURE_DOWNLOAD_TIMEOUT)
        data.readinto(f)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------
def _make_gcs_bucket(bucket_name: str):
    from google.cloud import storage
    return storage.Client().bucket(bucket_name)


def list_completed_basenames(gcs_bucket, gcs_prefix: str) -> Set[str]:
    """
    扫描 GCS 输出桶，返回已完成的全景图 basename 集合。
    判断标准: {gcs_prefix}/{basename}_front/metadata.json 存在。
    使用 delimiter 优化，只列出顶层 "目录" 再逐个检查 metadata.json。
    """
    completed: Set[str] = set()
    search_prefix = gcs_prefix.rstrip('/') + '/' if gcs_prefix else ''

    # 使用 delimiter='/' 只获取顶层"目录"前缀，避免遍历全部 blob
    for page in gcs_bucket.list_blobs(prefix=search_prefix, delimiter='/').pages:
        for prefix_str in page.prefixes:
            # prefix_str 格式: "output/IMG001_front/"
            dir_name = prefix_str.rstrip('/').rsplit('/', 1)[-1]
            if not dir_name.endswith('_front'):
                continue
            # 检查 metadata.json 是否存在
            meta_blob = gcs_bucket.blob(f"{prefix_str}metadata.json")
            if meta_blob.exists():
                completed.add(dir_name[:-6])  # 去掉 _front 后缀

    logging.info("GCS 上已完成 %d 张全景图", len(completed))
    return completed


def _upload_file_with_retry(gcs_bucket, local_path: str, gcs_path: str) -> None:
    """上传单个文件到 GCS，带重试"""
    for attempt in range(1, UPLOAD_MAX_RETRIES + 1):
        try:
            blob = gcs_bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            return
        except Exception as e:
            if attempt == UPLOAD_MAX_RETRIES:
                raise
            logging.warning("GCS 上传重试 %d/%d: %s - %s", attempt, UPLOAD_MAX_RETRIES, gcs_path, e)
            time.sleep(UPLOAD_RETRY_DELAY * attempt)


def upload_panorama_views(gcs_bucket, output_dir: str, basename: str, gcs_prefix: str) -> int:
    """只上传当前全景图的 3 个视角子目录到 GCS，返回上传文件数"""
    count = 0
    output_path = Path(output_dir)
    for view in ('left', 'front', 'right'):
        view_dir = output_path / f"{basename}_{view}"
        if not view_dir.is_dir():
            continue
        for fpath in view_dir.rglob('*'):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(output_path)
            gcs_path = f"{gcs_prefix.rstrip('/')}/{rel}"
            _upload_file_with_retry(gcs_bucket, str(fpath), gcs_path)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def cloud_batch_process(
    azure_client,
    gcs_bucket,
    gcs_prefix: str,
    azure_prefix: str = '',
    local_buffer: str = '/tmp/ai_city_buffer',
    limit: int = 0,
    config_overrides: Optional[dict] = None,
    force_refresh: bool = False,
    num_workers: int = 4,
    batch_size: int = 100,
    io_threads: int = 16,
) -> dict:
    """
    批量模式: 批量下载 → 批量处理 → 批量上传 → 清理，每批 batch_size 张。

    阶段分离的好处:
    - 下载阶段: io_threads 个线程并行下载，充分利用带宽
    - 处理阶段: num_workers 个 worker 并行处理，CPU/GPU 全力运行
    - 上传阶段: io_threads 个线程并行上传，不抢 CPU/GPU

    返回: {'success': int, 'fail': int, 'skipped': int, 'interrupted': bool}
    """
    # --- 列出待处理图片 ---
    all_images = list_azure_images(azure_client, azure_prefix, force_refresh=force_refresh)
    completed = list_completed_basenames(gcs_bucket, gcs_prefix)

    pending: List[str] = []
    for blob_name in all_images:
        basename = Path(blob_name).stem
        if basename not in completed:
            pending.append(blob_name)

    if limit > 0:
        pending = pending[:limit]

    total = len(pending)
    skip_count = len(completed)
    logging.info("待处理: %d 张 (已跳过: %d, batch=%d, workers=%d, io_threads=%d)",
                 total, skip_count, batch_size, num_workers, io_threads)

    if total == 0:
        logging.info("没有需要处理的图片，退出")
        return {'success': 0, 'fail': 0, 'skipped': skip_count, 'interrupted': False}

    # --- 创建目录 ---
    buffer_path = Path(local_buffer)
    input_dir = buffer_path / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)
    for wid in range(num_workers):
        (buffer_path / f'w{wid}' / 'output').mkdir(parents=True, exist_ok=True)

    base_config = get_default_config()
    if config_overrides:
        base_config.update(config_overrides)

    success_count = 0
    fail_count = 0
    start_time = time.time()

    # 静默 pipeline 的 print 输出
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    pbar = tqdm(total=total, desc="Processing", unit="img", file=sys.stderr)

    try:
        # --- 分批处理 ---
        for batch_start in range(0, total, batch_size):
            if _shutdown_event.is_set():
                break

            batch = pending[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            logging.info("=== 批次 %d/%d (%d 张) ===", batch_num, total_batches, len(batch))

            # ── Phase 1: 批量下载 ──
            local_files = {}  # blob_name -> local_path
            dl_errors = []

            def _download_one(blob_name: str) -> Tuple[str, Optional[str], str]:
                if _shutdown_event.is_set():
                    return blob_name, None, 'interrupted'
                local_path = str(input_dir / os.path.basename(blob_name))
                try:
                    download_blob(azure_client, blob_name, local_path)
                    return blob_name, local_path, ''
                except Exception as e:
                    return blob_name, None, str(e)

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=io_threads) as dl_pool:
                for blob_name, local_path, err in dl_pool.map(_download_one, batch):
                    if local_path:
                        local_files[blob_name] = local_path
                    else:
                        dl_errors.append((blob_name, err))
            logging.info("下载完成: %d 成功, %d 失败, %.1fs",
                         len(local_files), len(dl_errors), time.time() - t0)

            # 下载失败的直接记录
            for blob_name, err in dl_errors:
                fail_count += 1
                pbar.update(1)
                logging.error("FAIL(dl): %s - %s", Path(blob_name).stem, err)

            if _shutdown_event.is_set():
                break

            # ── Phase 2: 批量处理 (GPU + CPU) ──
            process_results = {}  # basename -> (ok, msg, wid)

            def _process_one(args_tuple) -> Tuple[str, bool, str, int]:
                blob_name, local_path, wid = args_tuple
                basename = Path(blob_name).stem
                w_output = buffer_path / f'w{wid}' / 'output'
                try:
                    config = copy.deepcopy(base_config)
                    result = process_panorama(local_path, str(w_output), config)
                    if not result.get('success'):
                        return basename, False, result.get('error', 'unknown'), wid
                    return basename, True, '', wid
                except Exception as e:
                    return basename, False, str(e), wid

            # 为每个图片分配 worker id (循环使用)
            process_tasks = []
            for i, (blob_name, local_path) in enumerate(local_files.items()):
                wid = i % num_workers
                process_tasks.append((blob_name, local_path, wid))

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=num_workers) as proc_pool:
                for basename, ok, msg, wid in proc_pool.map(_process_one, process_tasks):
                    process_results[basename] = (ok, msg, wid)
                    if not ok:
                        fail_count += 1
                        pbar.update(1)
                        pbar.set_postfix_str(f"ok={success_count} fail={fail_count}")
                        logging.error("FAIL(proc): %s - %s", basename, msg)
            logging.info("处理完成: %d 成功, %d 失败, %.1fs",
                         sum(1 for ok, _, _ in process_results.values() if ok),
                         sum(1 for ok, _, _ in process_results.values() if not ok),
                         time.time() - t0)

            # 清理输入文件 (处理完了就不需要了)
            for local_path in local_files.values():
                with contextlib.suppress(OSError):
                    os.unlink(local_path)

            if _shutdown_event.is_set():
                break

            # ── Phase 3: 批量上传 ──
            to_upload = [(bn, wid) for bn, (ok, _, wid) in process_results.items() if ok]

            def _upload_one(args_tuple) -> Tuple[str, bool, str, int]:
                basename, wid = args_tuple
                w_output = str(buffer_path / f'w{wid}' / 'output')
                try:
                    n = upload_panorama_views(gcs_bucket, w_output, basename, gcs_prefix)
                    return basename, True, f"{n} files", wid
                except Exception as e:
                    return basename, False, str(e), wid

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=io_threads) as ul_pool:
                for basename, ok, msg, wid in ul_pool.map(_upload_one, to_upload):
                    if ok:
                        success_count += 1
                        logging.info("OK: %s (%s)", basename, msg)
                    else:
                        fail_count += 1
                        logging.error("FAIL(ul): %s - %s", basename, msg)
                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={success_count} fail={fail_count}")
            logging.info("上传完成: %.1fs", time.time() - t0)

            # ── Phase 4: 清理输出文件 ──
            for basename, (ok, _, wid) in process_results.items():
                w_output = buffer_path / f'w{wid}' / 'output'
                for view in ('left', 'front', 'right'):
                    vd = w_output / f"{basename}_{view}"
                    if vd.exists():
                        shutil.rmtree(vd, ignore_errors=True)

    finally:
        pbar.close()
        sys.stdout.close()
        sys.stdout = old_stdout

    # --- 汇总 ---
    total_time = time.time() - start_time
    interrupted = _shutdown_event.is_set()

    logging.info("=" * 60)
    logging.info("批量处理%s", "被中断" if interrupted else "完成")
    logging.info("  之前已完成: %d", skip_count)
    logging.info("  本次成功: %d", success_count)
    logging.info("  本次失败: %d", fail_count)
    logging.info("  耗时: %.0fs (%.1f hr)", total_time, total_time / 3600)
    if success_count > 0:
        logging.info("  平均: %.1fs/张", total_time / success_count)
        logging.info("  速率: %.0f 张/hr", success_count / total_time * 3600)
    logging.info("=" * 60)

    return {
        'success': success_count,
        'fail': fail_count,
        'skipped': skip_count,
        'interrupted': interrupted,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    # 注册信号处理 (仅在作为主程序运行时，避免 import 时副作用)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(
        description='Cloud Batch Runner: Azure Blob → GCP VM → GCS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Azure 参数
    az = parser.add_argument_group('Azure Blob Storage')
    az.add_argument(
        '--azure-conn-str', default=os.environ.get('AZURE_STORAGE_CONNECTION_STRING', ''),
        help='Azure Storage 连接字符串 (或设置 AZURE_STORAGE_CONNECTION_STRING)',
    )
    az.add_argument('--azure-container', default='', help='Azure Blob 容器名')
    az.add_argument(
        '--azure-sas-url', default='',
        help='Azure Blob SAS URL (与 conn-str+container 二选一)',
    )
    az.add_argument('--azure-prefix', default='', help='仅处理此前缀下的 blob')

    # GCS 参数
    gc = parser.add_argument_group('Google Cloud Storage')
    gc.add_argument('--gcs-bucket', required=True, help='GCS 输出桶名称')
    gc.add_argument('--gcs-prefix', default='output', help='GCS 输出前缀 (默认: output)')

    # 处理参数
    proc = parser.add_argument_group('Processing')
    proc.add_argument('--local-buffer', default='/tmp/ai_city_buffer', help='本地缓冲目录')
    proc.add_argument('--limit', type=int, default=0, help='最多处理几张 (0=全部)')
    proc.add_argument('--depth-res', type=int, default=672, help='深度处理分辨率 (504/672/1008)')
    proc.add_argument('--png-compression', type=int, default=6, help='PNG 压缩等级 0-9 (默认 6)')
    proc.add_argument('--workers', type=int, default=4, help='并行处理 worker 数 (默认 4)')
    proc.add_argument('--batch-size', type=int, default=100,
                       help='每批下载/处理/上传的图片数 (默认 100)')
    proc.add_argument('--io-threads', type=int, default=16,
                       help='下载/上传并行线程数 (默认 16)')
    proc.add_argument('--gpu-concurrency', type=int, default=2,
                       help='GPU 并发推理数 (默认 2, 24GB GPU 可设 3)')
    proc.add_argument('--refresh-cache', action='store_true', help='强制重新列出 Azure blob (忽略缓存)')

    args = parser.parse_args()

    # --- 日志 ---
    log_file = Path(__file__).parent / 'cloud_batch_run.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(str(log_file), encoding='utf-8'),
        ],
    )

    # --- 初始化云客户端 ---
    logging.info("初始化云存储连接...")

    azure_client = _make_azure_client(
        args.azure_conn_str, args.azure_sas_url, args.azure_container,
    )
    gcs_bucket = _make_gcs_bucket(args.gcs_bucket)

    # --- 配置覆盖 ---
    config_overrides = {
        'depth_process_res': args.depth_res,
        'png_compression': args.png_compression,
    }

    # --- GPU 并发 ---
    set_gpu_concurrency(args.gpu_concurrency)
    logging.info("GPU concurrency: %d", args.gpu_concurrency)

    # --- 开跑 ---
    result = cloud_batch_process(
        azure_client=azure_client,
        gcs_bucket=gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        azure_prefix=args.azure_prefix,
        local_buffer=args.local_buffer,
        limit=args.limit,
        config_overrides=config_overrides,
        force_refresh=args.refresh_cache,
        num_workers=args.workers,
        batch_size=args.batch_size,
        io_threads=args.io_threads,
    )

    sys.exit(0 if not result['interrupted'] else 1)


if __name__ == '__main__':
    main()
