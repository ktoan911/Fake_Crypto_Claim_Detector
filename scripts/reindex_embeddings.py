"""Rebuild index: re-embed toàn corpus sang index MỚI (engine lucene) rồi
alias-swap để tên cũ trỏ sang index mới — fix cả bug embedding LẪN engine.

Vì sao:
  - Vector cũ sinh sai (AutoModel + mean-pool, không normalize) lệch pha với
    query (SentenceTransformer: CLS + Normalize). => re-embed bằng đúng pipeline.
  - Engine knn_vector cố định lúc tạo index; `nmslib` không hỗ trợ filtered-kNN
    (lỗi 400 -> fallback brute-force). Muốn `lucene` phải DỰNG INDEX MỚI.
  - OpenSearch KHÔNG có lệnh rename. "Đổi tên" = xoá index cũ rồi tạo ALIAS
    tên cũ trỏ vào index mới. App gọi tên cũ vẫn chạy, không phải sửa .env.

Luồng:
  1) tạo index mới (engine lucene) — mặc định {source}_v2;
  2) scroll toàn bộ doc nguồn -> encode lại content -> streaming_bulk (retry/backoff) vào index mới;
  3) kiểm tra số lượng khớp;
  4) (trừ khi --no-swap) xoá index nguồn, tạo alias {source} -> index mới.

Tối ưu GPU H200: encode fp16 + batch lớn; upload song song (nút cổ chai là
mạng tới OpenSearch managed, không phải GPU).

  Thử:   python scripts/reindex_embeddings.py --limit 500 --no-swap
  Full:  python scripts/reindex_embeddings.py --device cuda --encode-batch 512
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from opensearchpy.helpers import streaming_bulk
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.opensearch import OpenSearchKB  # noqa: E402

load_dotenv()


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild index (lucene) + re-embed + alias-swap, tối ưu GPU."
    )
    p.add_argument("--source-index", default=os.getenv("OP_KB_NAME"),
                   help="Index/alias nguồn (mặc định OP_KB_NAME).")
    p.add_argument("--new-index", default=None,
                   help="Index mới (mặc định {source}_v2). PHẢI khác source.")
    p.add_argument("--model", default=os.getenv("RETRIEVER_MODEL", "AITeamVN/Vietnamese_Embedding"))
    p.add_argument("--device", default=_default_device(), help="cuda | cpu.")
    p.add_argument("--encode-batch", type=int, default=384,
                   help="Batch encode GPU. H200 kéo được 256-1024.")
    p.add_argument("--block-size", type=int, default=6000,
                   help="Số doc gom mỗi vòng scan->encode->upload.")
    p.add_argument("--chunk-size", type=int, default=200,
                   help="Doc mỗi request bulk (nhỏ hơn = nhẹ tải cluster hơn).")
    p.add_argument("--max-retries", type=int, default=6,
                   help="Số lần tự retry mỗi doc khi cluster từ chối (429).")
    p.add_argument("--initial-backoff", type=float, default=2.0,
                   help="Giây chờ trước retry đầu (tăng gấp đôi mỗi lần).")
    p.add_argument("--limit", type=int, default=0, help="Chỉ xử lý N doc đầu (0=tất cả).")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=None,
                   help="Ép fp16 (mặc định bật khi cuda).")
    p.add_argument("--fp32", dest="fp16", action="store_false", help="Tắt fp16.")
    p.add_argument("--no-swap", action="store_true",
                   help="Chỉ build index mới, KHÔNG xoá cũ/alias — để tự kiểm tra.")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ đếm + encode, không tạo index/ghi/swap.")
    return p.parse_args()


def resolve_source_index(client, name: str) -> str:
    """Nếu `name` là alias, trả về index thật phía sau để có thể xoá đúng."""
    try:
        info = client.indices.get_alias(name=name)
        # {real_index: {"aliases": {name: {}}}}
        return next(iter(info.keys()))
    except Exception:
        return name


def main() -> int:
    args = parse_args()
    src_name = args.source_index
    if not src_name:
        print("[reindex] Thiếu source index (OP_KB_NAME).", file=sys.stderr)
        return 2
    new_index = args.new_index or f"{src_name}_v2"
    if new_index == src_name:
        print("[reindex] --new-index phải KHÁC source.", file=sys.stderr)
        return 2

    use_fp16 = args.fp16 if args.fp16 is not None else (args.device == "cuda")
    print(f"[reindex] REBUILD source={src_name} -> new={new_index} | model={args.model} "
          f"| device={args.device} | fp16={use_fp16} | encode_batch={args.encode_batch} "
          f"| block={args.block_size} | chunk={args.chunk_size} "
          f"| retries={args.max_retries} backoff={args.initial_backoff}s "
          f"| limit={args.limit or 'ALL'} | no_swap={args.no_swap} | dry_run={args.dry_run}")

    src = OpenSearchKB(index_name=src_name)
    real_src = resolve_source_index(src.client, src_name)
    if real_src != src_name:
        print(f"[reindex] '{src_name}' là alias -> index thật: {real_src}")
    total = src.count_docs()
    print(f"[reindex] tổng doc nguồn: {total}")

    embedder = SentenceTransformer(args.model, device=args.device)
    if use_fp16 and args.device == "cuda":
        embedder = embedder.half()
    dim = int(embedder.get_sentence_embedding_dimension())
    print(f"[reindex] embedding_dim = {dim}")

    dst = OpenSearchKB(index_name=new_index, embedding_dim=dim)
    if not args.dry_run:
        dst.create_index(overwrite=True)  # engine lucene (xem opensearch.create_index)
        print(f"[reindex] đã tạo index mới {new_index} (engine lucene).")

    body = {"query": {"match_all": {}}, "_source": True}
    resp = src.client.search(index=src_name, body=body, scroll="10m", size=args.chunk_size)
    scroll_id = resp.get("_scroll_id")
    hits = resp["hits"]["hits"]

    processed = written = errors = skipped = 0
    t0 = time.perf_counter()
    block_docs: list[dict] = []
    block_texts: list[str] = []
    err_reasons: dict[str, int] = {}   # loại lỗi -> số lần
    err_samples: list[dict] = []       # vài info lỗi đầy đủ đầu tiên
    fail_path = os.path.join(os.path.dirname(__file__), "reindex_failed_ids.txt")
    fail_fh = None if args.dry_run else open(fail_path, "w")

    def process_block() -> None:
        nonlocal block_docs, block_texts, written, errors
        if not block_docs:
            return
        vecs = embedder.encode(
            block_texts, normalize_embeddings=True, batch_size=args.encode_batch,
            show_progress_bar=False, convert_to_numpy=True,
        )
        for d, v in zip(block_docs, vecs):
            d["_source"]["embedding"] = v.tolist()
        if not args.dry_run:
            actions = ({
                "_op_type": "index", "_index": new_index,
                "_id": d["_id"], "_source": d["_source"],
            } for d in block_docs)
            for ok, info in streaming_bulk(
                dst.client, actions,
                chunk_size=args.chunk_size,
                max_retries=args.max_retries,
                initial_backoff=args.initial_backoff,
                raise_on_error=False, raise_on_exception=False,
            ):
                if ok:
                    written += 1
                    continue
                errors += 1
                # info = {"index": {"_id":..., "status":..., "error": {"type","reason"...}}}
                action = info.get("index") or info.get("create") or next(iter(info.values()), {})
                _id = action.get("_id")
                err = action.get("error")
                if isinstance(err, dict):
                    reason = f"{err.get('type', '?')}: {str(err.get('reason', ''))[:120]}"
                else:
                    reason = str(err)[:120] or f"status={action.get('status')}"
                err_reasons[reason] = err_reasons.get(reason, 0) + 1
                if len(err_samples) < 3:
                    err_samples.append(info)
                    print(f"[reindex]   MẪU LỖI: {reason}")
                if fail_fh and _id:
                    fail_fh.write(f"{_id}\n")
        block_docs = []
        block_texts = []

    stop = False
    while hits and not stop:
        for hit in hits:
            if args.limit and processed >= args.limit:
                stop = True
                break
            content = str((hit.get("_source") or {}).get("content") or "").strip()
            processed += 1
            if not content:
                skipped += 1
                continue
            block_docs.append({"_id": hit["_id"], "_source": hit["_source"]})
            block_texts.append(content)
        if len(block_docs) >= args.block_size:
            process_block()
            rate = processed / max(1e-6, time.perf_counter() - t0)
            print(f"[reindex] {processed}/{total} | {rate:.0f} doc/s | ghi={written} | lỗi={errors}")
        if stop:
            break
        resp = src.client.scroll(scroll_id=scroll_id, scroll="10m")
        scroll_id = resp.get("_scroll_id")
        hits = resp["hits"]["hits"]

    process_block()
    try:
        src.client.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass
    if fail_fh:
        fail_fh.close()

    elapsed = time.perf_counter() - t0
    print(f"[reindex] ENCODE/GHI XONG | quét={processed} | ghi={written} | lỗi={errors} "
          f"| bỏ(rỗng)={skipped} | {elapsed:.1f}s | {processed / max(1e-6, elapsed):.0f} doc/s")
    if err_reasons:
        print("[reindex] PHÂN LOẠI LỖI (loại -> số lần):")
        for reason, cnt in sorted(err_reasons.items(), key=lambda x: -x[1]):
            print(f"[reindex]   {cnt:>7}  {reason}")
        print(f"[reindex] Danh sách _id hỏng đã ghi: {fail_path}")

    if args.dry_run:
        print("[reindex] dry-run: không tạo index/không swap.")
        return 0

    dst.client.indices.refresh(index=new_index)
    new_count = dst.count_docs()
    expected = processed - skipped
    print(f"[reindex] doc index mới: {new_count} (kỳ vọng ~{expected})")

    if args.no_swap:
        print(f"[reindex] --no-swap: giữ nguyên. Kiểm tra {new_index} rồi swap thủ công:")
        print(f"           DELETE {real_src}  +  alias {src_name} -> {new_index}")
        return 0

    if errors > 0 or new_count < expected:
        print(f"[reindex] ! CÓ LỖI hoặc thiếu doc (errors={errors}, "
              f"new={new_count} < expected={expected}). BỎ QUA swap để an toàn.", file=sys.stderr)
        print(f"[reindex]   Index mới vẫn còn ({new_index}); sửa xong swap tay hoặc chạy lại.")
        return 1

    # Swap: xoá index nguồn thật rồi tạo alias tên cũ -> index mới.
    print(f"[reindex] SWAP: xoá index nguồn '{real_src}' và trỏ alias '{src_name}' -> '{new_index}'")
    src.client.indices.delete(index=real_src)
    dst.client.indices.put_alias(index=new_index, name=src_name)
    print(f"[reindex] XONG. '{src_name}' giờ là alias -> '{new_index}' (engine lucene). "
          f"Không cần đổi .env, restart API là chạy vector mới + filtered-kNN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
