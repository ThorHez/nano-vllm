#!/usr/bin/env python3
"""
清理共享内存脚本

用于清理残留的共享内存，解决 FileExistsError 问题
"""

from multiprocessing import shared_memory
import os
import sys


def cleanup_shared_memory(name="nanovllm"):
    """清理指定名称的共享内存"""
    try:
        # 尝试连接到已存在的共享内存
        shm = shared_memory.SharedMemory(name=name)
        print(f"找到共享内存: {name}")

        # 关闭并删除
        shm.close()
        shm.unlink()
        print(f"✅ 成功清理共享内存: {name}")

    except FileNotFoundError:
        print(f"ℹ️  共享内存 '{name}' 不存在，无需清理")

    except PermissionError:
        print(f"❌ 权限不足，无法清理共享内存: {name}")
        print("请尝试使用 sudo 运行此脚本")
        return False

    except Exception as e:
        print(f"❌ 清理共享内存时出错: {e}")
        return False

    return True


def main():
    print("🧹 开始清理共享内存...")

    # 清理默认的共享内存
    success = cleanup_shared_memory("nanovllm")

    # 也可以清理其他可能的共享内存名称
    other_names = ["nanovllm_tmp", "nanovllm_backup"]
    for name in other_names:
        cleanup_shared_memory(name)

    if success:
        print("✅ 清理完成！现在可以重新运行您的程序了。")
    else:
        print("⚠️  清理过程中遇到一些问题，但可能不影响程序运行。")


if __name__ == "__main__":
    main()
