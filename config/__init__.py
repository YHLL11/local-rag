"""config 包：集中管理项目配置（密钥走 .env，可调参数走 config.yaml）。

对外主要暴露：
    from config.settings import Settings
    settings = Settings.load()
"""

from .settings import Settings  # noqa: F401
