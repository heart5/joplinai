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
# ## Context 类

# %%
def _find_root():
    """向上遍历父目录查找 rootfile 标记以定位项目根。

    与 func/first.getdirmain() 使用相同的 rootfile 标记模式，
    但自成一体——pathmagic 是自举模块，不能依赖 func.first。
    """
    here = Path(__file__).resolve().parent
    for candidate in [here] + list(here.parents):
        if (candidate / "rootfile").exists():
            return candidate
    return here


class Context:
    """上下文管理器，将项目目录临时加入 sys.path。

    使用 rootfile 标记定位项目根（与 func/first.getdirmain() 相同策略），
    然后添加根、src/、aimod/、func/ 以便跨包导入。
    """
    def __enter__(self):
        root = _find_root()
        for sub in ["", "src", "aimod", "func"]:
            d = str(root / sub) if sub else str(root)
            if d not in sys.path:
                sys.path.insert(0, d)

    @staticmethod
    def printsyspath():
        for p in sys.path:
            print(Path(p).resolve())

    def __exit__(self, *args):
        pass


context = Context  # 向后兼容别名


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
