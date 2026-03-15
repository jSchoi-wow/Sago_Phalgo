"""
KIS API 인증 정보 저장/불러오기
config/credentials.json 에 로컬 저장 (git 제외)
"""

import json
from pathlib import Path

_CRED_FILE = Path(__file__).parent / "credentials.json"


def save(app_key: str, app_secret: str, account_no: str, account_prod: str = "01"):
    data = {
        "app_key":     app_key,
        "app_secret":  app_secret,
        "account_no":  account_no,
        "account_prod": account_prod,
    }
    _CRED_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load() -> dict:
    """저장된 인증정보 반환. 없으면 빈 dict."""
    if not _CRED_FILE.exists():
        return {}
    try:
        return json.loads(_CRED_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def apply_to_config():
    """저장된 인증정보를 kis_config 모듈 변수에 반영."""
    import config.kis_config as cfg
    cred = load()
    if cred.get("app_key"):    cfg.APP_KEY      = cred["app_key"]
    if cred.get("app_secret"): cfg.APP_SECRET   = cred["app_secret"]
    if cred.get("account_no"): cfg.ACCOUNT_NO   = cred["account_no"]
    if cred.get("account_prod"): cfg.ACCOUNT_PROD = cred["account_prod"]
