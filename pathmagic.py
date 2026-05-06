# encoding:utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,-kernelspec,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
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
    """上下文管理器，临时将项目根目录加入 sys.path，确保项目内模块可被导入。"""

    def __enter__(self) -> None:
        syspathlst = [Path(p).resolve() for p in sys.path]
        for inpath in ["..", ".", "src"]:
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
