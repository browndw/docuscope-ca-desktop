import os
import sys
from pathlib import Path

import streamlit.web.cli as stcli
from pydantic_settings import BaseSettings


class StreamlitConfig(BaseSettings):
    browser_server_address: str = "localhost"


def resolve_path(path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.getcwd())
    return str(Path(base_path) / path)


if __name__ == "__main__":
    config = StreamlitConfig()
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("webapp/index.py"),
        f"--browser.serverAddress={config.browser_server_address}",
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=minimal",
        "--client.showSidebarNavigation=false",
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.enableXsrfProtection=false",
        "--server.enableCORS=false",
        "--runner.enforceSerializableSessionState=true",
        "--theme.primaryColor=#F63366",
        "--theme.backgroundColor=#FFFFFF",
        "--theme.secondaryBackgroundColor=#EBECF0",
        "--theme.textColor=#262730",
        '--theme.font=Source Sans Pro',
    ]
    sys.exit(stcli.main())
