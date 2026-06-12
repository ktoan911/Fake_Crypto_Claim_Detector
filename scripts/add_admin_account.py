"""
  python scripts/add_admin_account.py
  python scripts/add_admin_account.py --username admin --password s3cr3t
  python scripts/add_admin_account.py --list          # liệt kê tài khoản
  python scripts/add_admin_account.py --delete admin  # xoá tài khoản
"""

import argparse
import getpass
import os
import sys

import bcrypt
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection

load_dotenv()

_ADMIN_INDEX = "admin"


def _build_client() -> OpenSearch:
    return OpenSearch(
        hosts=[
            {
                "host": os.getenv("OP_HOST"),
                "port": int(os.getenv("OP_PORT", "9200")),
                "scheme": "https",
            }
        ],
        http_auth=(os.getenv("OP_AUTH_USERNAME"), os.getenv("OP_AUTH_PASSWORD")),
        verify_certs=True,
        http_compress=True,
        timeout=15,
        connection_class=RequestsHttpConnection,
    )


def _ensure_index(client: OpenSearch) -> None:
    if client.indices.exists(index=_ADMIN_INDEX):
        return
    client.indices.create(
        index=_ADMIN_INDEX,
        body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "username": {"type": "keyword"},
                    "password": {"type": "keyword"},
                }
            },
        },
    )
    print(f"[+] Đã tạo index '{_ADMIN_INDEX}'.")


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _find_by_username(client: OpenSearch, username: str) -> dict | None:
    resp = client.search(
        index=_ADMIN_INDEX,
        body={"size": 1, "query": {"term": {"username": username}}},
    )
    hits = resp["hits"]["hits"]
    return hits[0] if hits else None


def cmd_add(client: OpenSearch, username: str, plain_password: str) -> None:
    _ensure_index(client)
    hashed = _hash_password(plain_password)
    existing = _find_by_username(client, username)

    if existing:
        client.update(
            index=_ADMIN_INDEX,
            id=existing["_id"],
            body={"doc": {"password": hashed}},
            refresh=True,
        )
        print(f"[~] Đã cập nhật mật khẩu cho tài khoản '{username}'.")
    else:
        client.index(
            index=_ADMIN_INDEX,
            body={"username": username, "password": hashed},
            refresh=True,
        )
        print(f"[+] Đã thêm tài khoản '{username}'.")


def cmd_list(client: OpenSearch) -> None:
    if not client.indices.exists(index=_ADMIN_INDEX):
        print("Index 'admin' chưa tồn tại.")
        return
    resp = client.search(
        index=_ADMIN_INDEX,
        body={"size": 100, "query": {"match_all": {}}, "_source": ["username"]},
    )
    hits = resp["hits"]["hits"]
    if not hits:
        print("Không có tài khoản nào.")
        return
    print(f"{'ID':<26}  username")
    print("-" * 44)
    for h in hits:
        print(f"{h['_id']:<26}  {h['_source']['username']}")


def cmd_delete(client: OpenSearch, username: str) -> None:
    existing = _find_by_username(client, username)
    if not existing:
        print(f"Không tìm thấy tài khoản '{username}'.")
        return
    confirm = input(f"Xác nhận xoá tài khoản '{username}'? [y/N] ").strip().lower()
    if confirm != "y":
        print("Đã huỷ.")
        return
    client.delete(index=_ADMIN_INDEX, id=existing["_id"], refresh=True)
    print(f"[-] Đã xoá tài khoản '{username}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quản lý tài khoản admin")
    parser.add_argument("--username", "-u", help="Tên đăng nhập")
    # --password/-p bị loại bỏ: CLI args hiện thị trong ps/shell history — dùng getpass thay thế.
    parser.add_argument("--list", "-l", action="store_true", help="Liệt kê tài khoản")
    parser.add_argument("--delete", "-d", metavar="USERNAME", help="Xoá tài khoản")
    args = parser.parse_args()

    client = _build_client()

    if args.list:
        cmd_list(client)
        return

    if args.delete:
        cmd_delete(client, args.delete)
        return

    # Add / update mode
    username = args.username or input("Tên đăng nhập: ").strip()
    if not username:
        print("Tên đăng nhập không được để trống.")
        sys.exit(1)

    plain_password = getpass.getpass("Mật khẩu: ")
    confirm = getpass.getpass("Nhập lại mật khẩu: ")
    if plain_password != confirm:
        print("Mật khẩu không khớp.")
        sys.exit(1)

    if not plain_password:
        print("Mật khẩu không được để trống.")
        sys.exit(1)

    cmd_add(client, username, plain_password)


if __name__ == "__main__":
    main()
