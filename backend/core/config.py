from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    ENV_STATE = str | None


class Settings(BaseConfig):
    DATABASE_URL: str | None = None
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

    DATABASE_URL: str = "http://test.com"
    DB_FORCE_ROLL_BACK: bool = True


@lru_cache
def get_config(env_state: str):
    configs = {"dev": DevConfig, "prod": ProdConfig, "test": TestConfig}
    return configs[env_state]()


settings = get_config(BaseConfig().ENV_STATE)
