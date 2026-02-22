"""
AI City View — Gradio 前端
拖拽上传全景图，批量处理，实时显示进度和日志。
"""

import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr

from main import get_default_config, process_panorama


def _deduplicate_filename(name: str, seen: dict) -> str:
    """如果文件名重复，自动添加后缀 _2, _3, ..."""
    if name not in seen:
        seen[name] = 1
        return name
    seen[name] += 1
    stem = Path(name).stem
    suffix = Path(name).suffix
    return f"{stem}_{seen[name]}{suffix}"


def run_batch(files, output_dir, progress=gr.Progress(track_tqdm=False)):
    """
    批量处理上传的全景图（generator 模式，逐张 yield 更新进度和日志）。
    """
    if not files:
        yield "请先上传图片", "", 0
        return

    output_dir = output_dir.strip() or "output"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = get_default_config()
    total = len(files)
    log_lines = []
    success_count = 0
    fail_count = 0
    batch_start = time.time()
    seen_names: dict = {}

    for idx, file_obj in enumerate(files, 1):
        src = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
        original_name = src.name
        unique_name = _deduplicate_filename(original_name, seen_names)

        progress_text = f"进度: {idx}/{total} — 正在处理 {unique_name}"
        progress((idx - 1) / total, desc=progress_text)
        yield progress_text, "\n".join(log_lines), idx - 1

        # 复制到临时目录，避免影响原始文件
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_file = tmp_dir / unique_name
        try:
            shutil.copy2(str(src), str(tmp_file))
        except Exception as e:
            log_lines.append(f"[{idx}/{total}] ❌ {unique_name} — 复制失败: {e}")
            fail_count += 1
            continue

        t0 = time.time()
        try:
            result = process_panorama(str(tmp_file), str(output_path), config)
            elapsed = time.time() - t0
            if result.get("success"):
                log_lines.append(f"[{idx}/{total}] ✅ {unique_name} ({elapsed:.1f}s)")
                success_count += 1
            else:
                err = result.get("error", "未知错误")
                log_lines.append(f"[{idx}/{total}] ❌ {unique_name} — {err}")
                fail_count += 1
        except Exception as e:
            elapsed = time.time() - t0
            log_lines.append(f"[{idx}/{total}] ❌ {unique_name} — 异常: {e}")
            fail_count += 1
        finally:
            # 清理临时文件
            shutil.rmtree(tmp_dir, ignore_errors=True)

        progress(idx / total, desc=f"进度: {idx}/{total}")
        yield f"进度: {idx}/{total}", "\n".join(log_lines), idx

    # 最终统计
    batch_elapsed = time.time() - batch_start
    avg = batch_elapsed / total if total else 0
    summary = (
        f"\n{'='*40}\n"
        f"处理完成！\n"
        f"  成功: {success_count}  失败: {fail_count}  共: {total}\n"
        f"  总耗时: {batch_elapsed:.1f}s  平均: {avg:.1f}s/张\n"
        f"  输出目录: {output_path.resolve()}\n"
        f"{'='*40}"
    )
    log_lines.append(summary)
    yield f"完成 {success_count}/{total}", "\n".join(log_lines), total


# ── Gradio UI ──────────────────────────────────────────────

with gr.Blocks(title="AI City View") as demo:
    gr.Markdown("# AI City View 全景图批量处理工具")
    gr.Markdown("上传全景图 → 自动裁剪为 left/front/right 三个视角 → 每个视角生成 25 个分析文件")

    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="拖拽上传全景图（支持多选 jpg/jpeg/png）",
            )
            output_dir_input = gr.Textbox(
                value="output",
                label="输出目录",
                placeholder="默认: output",
            )
            run_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")

        with gr.Column(scale=2):
            status_text = gr.Textbox(label="状态", interactive=False)
            log_output = gr.Textbox(
                label="处理日志",
                interactive=False,
                lines=15,
                max_lines=30,
            )
            count_display = gr.Number(label="已完成数量", interactive=False, visible=False)

    run_btn.click(
        fn=run_batch,
        inputs=[file_input, output_dir_input],
        outputs=[status_text, log_output, count_display],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
