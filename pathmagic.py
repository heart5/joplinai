# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 魔法路径

# %% [markdown]
# ## 引入库

# %%
import sys
from pathlib import Path


# %% [markdown]
# ## context类

# %%
class context:  # noqa: N801
    """上下文管理器，临时将项目根目录和 src/ 加入 sys.path。"""

    def __enter__(self) -> None:
        syspathlst = [Path(p).resolve() for p in sys.path]
        for inpath in [".", "src"]:
            if Path(inpath).resolve() not in syspathlst:
                sys.path.append(inpath)

    @staticmethod
    def printsyspath() -> None:
        for pson in sys.path:
            print(Path(pson).resolve())
        print(10 * "*")

    def __exit__(self, *args: any) -> None:
        pass


# %% [markdown]
# ## 主函数main()

# %%
if __name__ == "__main__":
    if "__file__" in list(locals()):
        print(f"运行文件\t{__file__}")

    for pp in sys.path:
        pson = Path(pp).resolve()
        print(pson)
    print("Done.完毕。")
