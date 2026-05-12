"""Joplinai AI 核心模块：嵌入、向量数据库、缓存与数据中心客户端。"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """统一日志获取函数 — 返回配置好格式的 logger 实例。

    优先使用 func.logme.log（云端配置格式），降级时回退到标准 logging。
    """
    try:
        import pathmagic
        with pathmagic.Context():
            from func.logme import log as _log
        # func.logme.log 是模块级 logger，返回基于 name 的新 logger
        logger = logging.getLogger(name)
        for h in _log.handlers:
            logger.addHandler(h)
        logger.setLevel(_log.level)
        return logger
    except ImportError:
        pass

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
