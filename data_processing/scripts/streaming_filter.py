import zstandard as zstd
import json
import io
import os
import time


def open_text_stream(input_path):
    """
    自动检测文件类型：
    - zstd 压缩文件（魔术头 28 b5 2f fd）用 ZstdDecompressor 流式解压
    - 普通文本/ndjson 文件直接打开
    """
    with open(input_path, 'rb') as fh:
        magic = fh.read(4)

    is_zstd = magic == b'\x28\xb5\x2f\xfd'

    if is_zstd:
        print(f"[Detect] {input_path} is zstd compressed")
        fh = open(input_path, 'rb')
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        stream_reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream_reader, encoding='utf-8'), fh
    else:
        print(f"[Detect] {input_path} is plain text / ndjson")
        fh = open(input_path, 'r', encoding='utf-8')
        return fh, fh


def stream_filter(input_path, output_path, start_ts, end_ts, progress_interval=1_000_000):
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Please place the dataset at this path or update the path in the script."
        )

    # 2. 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    input_size_gb = os.path.getsize(input_path) / (1024 ** 3)
    print(f"[Start] Filtering {input_path}")
    print(f"[Start] Input size: {input_size_gb:.2f} GB")
    print(f"[Start] Time range: {start_ts} to {end_ts}")
    print(f"[Start] Output will be written to: {output_path}")

    total_lines = 0
    written_lines = 0
    start_time = time.time()

    text_stream, file_handle = open_text_stream(input_path)

    with text_stream:
        with open(output_path, 'w', encoding='utf-8') as out:
            try:
                for line in text_stream:
                    total_lines += 1

                    # 每处理 progress_interval 行打印一次进度
                    if total_lines % progress_interval == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"[Progress] Processed {total_lines:,} lines, "
                            f"written {written_lines:,} matching records "
                            f"({elapsed:.1f}s elapsed)"
                        )

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    created = obj.get('created_utc')
                    if created and start_ts <= int(created) <= end_ts:
                        out.write(line)
                        written_lines += 1

            except KeyboardInterrupt:
                print("\n[Interrupted] Stopping early due to user interrupt.")

    # 关闭底层文件句柄
    file_handle.close()

    elapsed = time.time() - start_time
    print(f"[Done] Processed {total_lines:,} lines in total")
    print(f"[Done] Wrote {written_lines:,} matching records to {output_path}")
    print(f"[Done] Time elapsed: {elapsed:.1f}s")

    if written_lines == 0:
        print("[Warning] No records matched the time range. Check your start_ts/end_ts.")

    print("-" * 60)


def stream_filter_years(input_path, output_dir, prefix, suffix, years, progress_interval=1_000_000):
    """
    一次流式读取，按年份把记录写入多个输出文件。
    years: [(year, start_ts, end_ts), ...]
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    os.makedirs(output_dir, exist_ok=True)

    input_size_gb = os.path.getsize(input_path) / (1024 ** 3)
    print(f"[Start] Filtering {input_path}")
    print(f"[Start] Input size: {input_size_gb:.2f} GB")
    print(f"[Start] Years: {[y for y, _, _ in years]}")

    # 打开所有输出文件
    year_map = {}
    file_handles = {}
    for year, start_ts, end_ts in years:
        output_path = os.path.join(output_dir, f"{prefix}{year}{suffix}")
        print(f"[Start] {year}: {start_ts} to {end_ts} -> {output_path}")
        file_handles[year] = open(output_path, 'w', encoding='utf-8')
        year_map[year] = (start_ts, end_ts, 0)

    total_lines = 0
    start_time = time.time()
    text_stream, input_fh = open_text_stream(input_path)

    try:
        with text_stream:
            for line in text_stream:
                total_lines += 1
                if total_lines % progress_interval == 0:
                    elapsed = time.time() - start_time
                    written_sum = sum(c for _, _, c in year_map.values())
                    print(
                        f"[Progress] Processed {total_lines:,} lines, "
                        f"written {written_sum:,} matching records "
                        f"({elapsed:.1f}s elapsed)"
                    )

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                created = obj.get('created_utc')
                if not created:
                    continue
                ts = int(created)
                for year, (start_ts, end_ts, count) in year_map.items():
                    if start_ts <= ts <= end_ts:
                        file_handles[year].write(line)
                        year_map[year] = (start_ts, end_ts, count + 1)
    finally:
        input_fh.close()
        for fh in file_handles.values():
            fh.close()

    elapsed = time.time() - start_time
    print(f"[Done] Processed {total_lines:,} lines in total")
    for year, (_, _, count) in year_map.items():
        print(f"[Done] Wrote {count:,} records for {year}")
        if count == 0:
            print(f"[Warning] No records matched year {year}.")
    print(f"[Done] Time elapsed: {elapsed:.1f}s")
    print("-" * 60)


if __name__ == "__main__":
    from datetime import datetime, timezone

    # 目标年份：2022, 2023, 2024, 2025
    target_years = [2022, 2023, 2024, 2025]
    year_ranges = []
    for year in target_years:
        start_ts = int(datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())
        year_ranges.append((year, start_ts, end_ts))

    output_dir = 'data_processing/reddit'

    # 处理 comments（zstd 压缩，8 GB）
    stream_filter_years(
        'data_processing/reddit/subreddits25/wallstreetbets_comments.zst',
        output_dir=output_dir,
        prefix='wallstreetbets_comments_',
        suffix='.ndjson',
        years=year_ranges
    )

    # 处理 submissions（zstd 压缩，~600 MB）
    stream_filter_years(
        'data_processing/reddit/subreddits25/wallstreetbets_submissions.zst',
        output_dir=output_dir,
        prefix='wallstreetbets_submissions_',
        suffix='.ndjson',
        years=year_ranges
    )
