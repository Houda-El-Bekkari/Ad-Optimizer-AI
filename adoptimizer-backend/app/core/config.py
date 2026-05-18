from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:chaymae2002@localhost:5433/adoptimizer"
    jwt_secret_key: str = "change-this-secret-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    chatbot_webhook_url: str = "http://localhost:5678/webhook-test/llm-chatbot"
    case1_audit_webhook_url: str = "http://localhost:5678/webhook/case1-optimization-campaign-existing"
    case1_dashboard_webhook_url: str = "http://localhost:5678/webhook/case1-optimization-dashboard"
    case2_campaign_webhook_url: str = "http://localhost:5678/webhook/case2-creation-nouvelle-campagne"
    cors_origins: str = "http://localhost:4200,http://127.0.0.1:4200"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
