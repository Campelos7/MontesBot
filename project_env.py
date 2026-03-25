"""
Carrega o ficheiro .env na raiz do repositório.

Sem isto, `load_dotenv()` sem caminho só procura o .env no diretório de trabalho
(current working directory). Se o utilizador corre o Uvicorn a partir de outra pasta,
as variáveis (GROQ_API_KEY, etc.) não são carregadas e o Groq devolve 401.
"""

from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"


def load_project_env() -> None:
    """Carrega .env da raiz do MontesBot, com override."""
    if ENV_FILE.is_file():
        load_dotenv(ENV_FILE, override=True)
