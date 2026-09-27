#!/usr/bin/env python3
"""Fail when source imports cross TramFlow's documented architecture boundaries."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend" / "app"
FRONTEND_ROOT = ROOT / "frontend" / "src"


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    message: str

    def render(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.message}"


FRAMEWORK_PREFIXES = ("fastapi", "sqlalchemy", "pydantic", "pydantic_settings")
FORBIDDEN_IMPORTS: dict[str, tuple[str, ...]] = {
    "domain": FRAMEWORK_PREFIXES
    + (
        "app.api",
        "app.application",
        "app.core",
        "app.infrastructure",
        "app.ml",
        "app.schemas",
    ),
    "application": FRAMEWORK_PREFIXES
    + ("app.api", "app.core", "app.infrastructure", "app.schemas"),
    "infrastructure": ("app.api", "app.schemas"),
    "ml": FRAMEWORK_PREFIXES + ("app.api", "app.infrastructure", "app.schemas"),
}
FETCH_PATTERN = re.compile(r"(?<![\w$])(?:globalThis\.|window\.)?fetch\s*\(")


def has_prefix(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def check_backend() -> list[Violation]:
    violations: list[Violation] = []
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        relative = path.relative_to(BACKEND_ROOT)
        layer = relative.parts[0] if len(relative.parts) > 1 else "root"
        forbidden = FORBIDDEN_IMPORTS.get(layer)
        if forbidden is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            violations.append(Violation(path, error.lineno or 1, f"cannot parse Python: {error.msg}"))
            continue
        for line, module in imported_modules(tree):
            if has_prefix(module, forbidden):
                violations.append(
                    Violation(path, line, f"{layer} layer must not import {module}")
                )
    return violations


def check_frontend() -> list[Violation]:
    violations: list[Violation] = []
    for pattern in ("*.ts", "*.tsx"):
        for path in sorted(FRONTEND_ROOT.rglob(pattern)):
            relative = path.relative_to(FRONTEND_ROOT)
            if relative.parts[0] in {"api", "test"}:
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if FETCH_PATTERN.search(line):
                    violations.append(
                        Violation(
                            path,
                            line_number,
                            "direct fetch is only allowed in frontend/src/api",
                        )
                    )
    return violations


def main() -> int:
    violations = check_backend() + check_frontend()
    if violations:
        print("Architecture boundary violations:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation.render()}", file=sys.stderr)
        return 1
    print("Architecture boundaries passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
