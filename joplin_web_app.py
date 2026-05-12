# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # 门户应用 — 入口文件

# %% [markdown]
# 实际实现在 `src/web_app/` 包中。

# %%
if __name__ == "__main__":
    import pathmagic
    with pathmagic.Context():
        from func.first import getdirmain
    template_dir = getdirmain() / "templates"
    template_dir.mkdir(exist_ok=True)
    from src.web_app import create_app
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=False)
