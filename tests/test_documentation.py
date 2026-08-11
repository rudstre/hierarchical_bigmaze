import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
MARKDOWN_FILES = (
    PROJECT_ROOT / "README.md",
    *sorted((PROJECT_ROOT / "docs").glob("*.md")),
)
MARKDOWN_LINK = re.compile(r"\[(?P<label>[^]]+)]\((?P<target>[^)]+)\)")
CODE_SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")


def _local_target(source: Path, target: str) -> Path | None:
    if target.startswith(("http://", "https://", "#")):
        return None
    relative_path = target.split("#", maxsplit=1)[0]
    return (source.parent / relative_path).resolve()


def test_project_markdown_links_resolve():
    for source in MARKDOWN_FILES:
        markdown = source.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(markdown):
            target = _local_target(source, match.group("target"))
            if target is not None:
                assert target.exists(), (
                    f"{source.relative_to(PROJECT_ROOT)} links to missing "
                    f"{match.group('target')}"
                )


def test_python_source_links_name_symbols_in_their_target():
    for source in MARKDOWN_FILES:
        markdown = source.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(markdown):
            target = _local_target(source, match.group("target"))
            if target is None or target.suffix != ".py":
                continue
            python_source = target.read_text(encoding="utf-8")
            for qualified_name in CODE_SYMBOL.findall(match.group("label")):
                symbol = qualified_name.rsplit(".", maxsplit=1)[-1]
                declaration = re.compile(
                    rf"^\s*(?:class|def)\s+{re.escape(symbol)}\b",
                    re.MULTILINE,
                )
                assert declaration.search(python_source), (
                    f"{source.relative_to(PROJECT_ROOT)} names "
                    f"{qualified_name}, which is not declared in "
                    f"{target.relative_to(PROJECT_ROOT)}"
                )
