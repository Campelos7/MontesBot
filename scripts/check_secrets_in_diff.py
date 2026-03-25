#!/usr/bin/env python3
"""
Falha com código de saída != 0 se o diff entre dois commits introduzir
atribuições suspeitas de chaves de API (ex.: GROQ_API_KEY=valor real).

Uso:
  python scripts/check_secrets_in_diff.py <base_sha> <head_sha>
  python scripts/check_secrets_in_diff.py   # local: merge-base..HEAD, ou diff em stage (git add)
"""
from __future__ import annotations

import re
import subprocess
import sys

# Nomes de variáveis que não devem aparecer no diff com valores credenciais.
SECRET_VAR_NAMES = (
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "SCRAPE_ADMIN_TOKEN",
)

# Valores claramente placeholders (comparação case-insensitive).
PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "tua_chave_aqui",
        "your_key_here",
        "changeme",
        "placeholder",
        "xxx",
        "redacted",
        "sk-xxx",
        "sk-xxxxx",
        # Exemplo no .env.example do projeto (não é segredo real)
        "token_secreto_para_manual_scrape",
    }
)

# Atribuição estilo .env ou JSON: VAR=... ou "VAR": "..."
_ASSIGN_PATTERNS = [
    re.compile(
        rf'(?P<name>{"|".join(re.escape(n) for n in SECRET_VAR_NAMES)})\s*=\s*(?P<val>[^\s#]+)',
        re.IGNORECASE,
    ),
    re.compile(
        rf'["\']?(?P<name>{"|".join(re.escape(n) for n in SECRET_VAR_NAMES)})["\']?\s*:\s*["\']?(?P<val>[^"\'\s#,}}]+)',
        re.IGNORECASE,
    ),
]

# Chaves Groq reais começam por gsk_; padrão genérico longo após o nome da var.
_GROQ_INLINE = re.compile(r"gsk_[a-zA-Z0-9]{20,}")
_SK_OPENAI = re.compile(r"sk-(?:proj-)?[a-zA-Z0-9]{20,}")


def _is_placeholder(val: str) -> bool:
    v = val.strip().strip('"').strip("'")
    if not v:
        return True
    if v.lower() in PLACEHOLDER_VALUES:
        return True
    if v.startswith("${") or v.startswith("<"):
        return True
    if v in ("...", "…"):
        return True
    return False


def _strip_hash_comment(line: str) -> str:
    """Remove comentário estilo shell/Python a partir do primeiro # não dentro de aspas (heurística simples)."""
    if "#" not in line:
        return line
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i].rstrip()
    return line


def _check_assignment_line(line: str) -> list[str]:
    problems: list[str] = []
    work = _strip_hash_comment(line)
    for pat in _ASSIGN_PATTERNS:
        m = pat.search(work)
        if not m:
            continue
        val = m.group("val")
        if _is_placeholder(val):
            continue
        # Valor curto tipo "token" no .env.example ainda pode ser placeholder
        if len(val) <= 6 and val.isalpha():
            continue
        problems.append(f"atribuição suspeita: {m.group('name')}=… (valor não é placeholder óbvio)")
    if _GROQ_INLINE.search(work):
        problems.append("possível chave Groq (prefixo gsk_) no diff")
    if _SK_OPENAI.search(work) and "sk-xxx" not in work.lower():
        problems.append("possível chave estilo OpenAI (sk-…) no diff")
    return problems


def _parse_diff_added_lines(diff_stdout: str) -> list[str]:
    added: list[str] = []
    for line in diff_stdout.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added.append(line[1:])
    return added


def _git_diff_added_lines(base: str, head: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "-U0", "--no-color", base, head],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0 and out.stderr:
        print(out.stderr, file=sys.stderr)
    return _parse_diff_added_lines(out.stdout)


def _staged_diff_added_lines() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--no-color"],
        capture_output=True,
        text=True,
        check=False,
    )
    return _parse_diff_added_lines(out.stdout)


def _resolve_base_head(argv: list[str]) -> tuple[str, str]:
    if len(argv) >= 2:
        return argv[0], argv[1]
    # Repositório novo: um único commit
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for ref in ("origin/main", "origin/master", "main", "master"):
        try:
            base = subprocess.check_output(
                ["git", "merge-base", head, ref],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return base, head
        except subprocess.CalledProcessError:
            continue
    try:
        base = subprocess.check_output(["git", "rev-parse", "HEAD~1"], text=True).strip()
    except subprocess.CalledProcessError:
        base = head
    return base, head


_GROQ_STRICT = re.compile(
    r"GROQ_API_KEY\s*=\s*(\S+)",
    re.IGNORECASE,
)


def _collect_failures(added: list[str]) -> list[str]:
    by_line: dict[str, list[str]] = {}
    for raw in added:
        line_full = raw.rstrip()
        for msg in _check_assignment_line(line_full):
            by_line.setdefault(line_full, []).append(msg)

    for raw in added:
        line_full = raw.rstrip()
        work = _strip_hash_comment(line_full)
        m = _GROQ_STRICT.search(work)
        if not m:
            continue
        val = m.group(1).strip().strip('"').strip("'")
        if val.lower() == "tua_chave_aqui":
            continue
        if _is_placeholder(val):
            continue
        by_line.setdefault(line_full, []).append(
            "GROQ_API_KEY com valor que não é o placeholder do .env.example"
        )

    failures: list[str] = []
    for line, msgs in by_line.items():
        unique = list(dict.fromkeys(msgs))
        failures.append("  - " + "; ".join(unique) + "\n    " + line[:200])
    return failures


def main() -> int:
    base, head = _resolve_base_head(sys.argv[1:])
    if base != head:
        added = _git_diff_added_lines(base, head)
        label = "%s..%s" % (base[:7], head[:7])
    else:
        added = _staged_diff_added_lines()
        if not added:
            print(
                "check_secrets_in_diff: sem commits anteriores úteis nem alterações em stage; "
                "nada a verificar. (Antes do commit: git add … e corre de novo, ou passa dois SHAs.)"
            )
            return 0
        label = "staged"

    failures = _collect_failures(added)

    if failures:
        print(
            "ERRO: o diff introduz possíveis segredos ou atribuições de GROQ_API_KEY não permitidas.\n",
            file=sys.stderr,
        )
        print("\n".join(failures), file=sys.stderr)
        return 1

    print("check_secrets_in_diff: OK (%s)." % label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
