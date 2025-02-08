from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ENV_STATE: str | None = None


class Settings(BaseConfig):
    DATABASE_URL: str | None = None
    DEEPSEEK_API_KEY: str | None = None
    BASE_URL: str | None = None
    DB_FORCE_ROLL_BACK: bool = False
    OPENAI_API_KEY: str | None = None
    WEATHER_API_KEY: str | None = None
    GEO_API_KEY: str | None = None


class DevConfig(Settings):
    model_config = SettingsConfigDict(env_prefix="DEV_", extra="ignore")


class ProdConfig(Settings):
    model_config = SettingsConfigDict(env_prefix="PROD_", extra="ignore")


class TestConfig(Settings):
    model_config = SettingsConfigDict(env_prefix="TEST_", extra="ignore")
    DATABASE_URL: str = "postgresql://paydanticai:paydanticai@localhost/testdb"
    DB_FORCE_ROLL_BACK: bool = True


# Avoid reloading environment variables multiple times
@lru_cache
def get_config(env_state: str):
    configs = {"dev": DevConfig, "prod": ProdConfig, "test": TestConfig}
    return configs.get(env_state, DevConfig)()


settings = get_config(BaseConfig().ENV_STATE)
