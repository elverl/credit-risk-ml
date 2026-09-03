"""Deterministic, Markdown-aware chunking for the V2 knowledge base.

Chunk boundaries are driven by document structure and semantic units.  There
is intentionally no character- or token-count based splitting in this module.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_METADATA = ("kb_version", "model_version", "status")
FEATURE_SECTION = "Features en orden contractual"


def _slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "section"


def _front_matter(markdown: str) -> tuple[dict[str, object], str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Knowledge-base document is missing YAML front matter")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("Unclosed YAML front matter") from exc
    metadata: dict[str, object] = {}
    for line in lines[1:closing]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Unsupported front-matter line: {line!r}")
        clean_value: object = value.strip().strip('"\'')
        if isinstance(clean_value, str) and clean_value.isdigit():
            clean_value = int(clean_value)
        metadata[key.strip()] = clean_value
    missing = set(REQUIRED_METADATA) - metadata.keys()
    if missing:
        raise ValueError(f"Missing metadata: {sorted(missing)}")
    return metadata, "\n".join(lines[closing + 1 :]).strip()


def _sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Return the H1 title and ordered H2 sections with their original text."""
    title = ""
    current = ""
    h1_content: list[str] = []
    buffers: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    for line in body.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("## "):
            current = line[3:].strip()
            order.append(current)
        elif current:
            buffers[current].append(line)
        elif title:
            h1_content.append(line)
    if not order and title and "\n".join(h1_content).strip():
        return title, [(title, "\n".join(h1_content).strip())]
    return title, [(heading, "\n".join(buffers[heading]).strip()) for heading in order]


def _table_parts(content: str) -> tuple[list[str], list[str], list[str]]:
    prefix: list[str] = []
    header: list[str] = []
    rows: list[str] = []
    for line in content.splitlines():
        if line.startswith("|"):
            if len(header) < 2:
                header.append(line)
            else:
                rows.append(line)
        elif not header:
            prefix.append(line)
    return prefix, header, rows


def _paragraphs(content: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]


def _make_chunk(
    source: str,
    doc_type: str,
    metadata: dict[str, object],
    title: str,
    section: str,
    semantic_name: str,
    text: str,
    tags: Iterable[str],
    **extra: str,
) -> dict[str, object]:
    section_path = " > ".join(part for part in (title, section, semantic_name) if part)
    identifier_parts = [_slug(Path(source).stem), _slug(section)]
    if semantic_name:
        identifier_parts.append(_slug(semantic_name))
    chunk = {
        "chunk_id": "kb1-" + "--".join(identifier_parts),
        "source": source,
        "section_path": section_path,
        "doc_type": doc_type,
        "kb_version": metadata["kb_version"],
        "model_version": metadata["model_version"],
        "status": metadata["status"],
        "text": text.strip(),
        "tags": list(dict.fromkeys([doc_type, _slug(section), *tags])),
    }
    chunk.update(extra)
    return chunk


def _data_dictionary_chunks(
    source: str, metadata: dict[str, object], title: str, sections: list[tuple[str, str]]
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for section, content in sections:
        if section == FEATURE_SECTION:
            _, header, rows = _table_parts(content)
            for row in rows:
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                feature = cells[1].strip("`")
                text = f"## {section}\n\n" + "\n".join([*header, row])
                chunks.append(_make_chunk(source, "data_dictionary", metadata, title,
                                          section, feature, text, ["feature"],
                                          feature_name=feature))
        elif section == "Target":
            _, header, rows = _table_parts(content)
            for row in rows:
                target = row.strip("|").split("|")[0].strip().strip("`")
                text = f"## {section}\n\n" + "\n".join([*header, row])
                chunks.append(_make_chunk(source, "data_dictionary", metadata, title,
                                          section, target, text, ["target"],
                                          target_name=target))
        else:
            chunks.append(_make_chunk(source, "data_dictionary", metadata, title,
                                      section, "", f"## {section}\n\n{content}", []))
    return chunks


def _methodology_chunks(
    source: str, metadata: dict[str, object], title: str, sections: list[tuple[str, str]]
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for section, content in sections:
        if section == "Datos y split":
            chunks.append(_make_chunk(source, "modeling_methodology", metadata, title,
                                      section, "train-test", f"## {section}\n\n{content}",
                                      ["train", "test", "split"]))
        elif section == "OOF y selección de threshold":
            parts = _paragraphs(content)
            steps = parts[1].splitlines() if len(parts) > 1 else []
            groups = [
                ("oof", [*steps[:2]], ["oof", "train"]),
                ("youden", steps[2:4], ["youden", "threshold"]),
                ("threshold", [*steps[3:], *parts[2:]], ["threshold", "test"]),
            ]
            for name, lines, tags in groups:
                selected = [parts[0], *lines]
                chunks.append(_make_chunk(source, "modeling_methodology", metadata, title,
                                          section, name, f"## {section}\n\n" + "\n".join(selected), tags))
        elif section == "Selección del champion":
            parts = _paragraphs(content)
            chunks.append(_make_chunk(source, "modeling_methodology", metadata, title,
                                      section, "metricas", f"## {section}\n\n{parts[0]}",
                                      ["metrics", "model-comparison"]))
            chunks.append(_make_chunk(source, "modeling_methodology", metadata, title,
                                      section, "champion-selection",
                                      f"## {section}\n\n" + "\n\n".join(parts[1:]),
                                      ["champion", "selection"]))
        else:
            chunks.append(_make_chunk(source, "modeling_methodology", metadata, title,
                                      section, "", f"## {section}\n\n{content}", []))
    return chunks


def _shap_chunks(
    source: str, metadata: dict[str, object], title: str, sections: list[tuple[str, str]]
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for section, content in sections:
        if section == "Alcance":
            parts = _paragraphs(content)
            intro = parts[0]
            bullets = parts[1].splitlines()
            causal = parts[2]
            units = [
                ("interpretacion", intro, ["shap", "interpretation"]),
                ("signo-positivo", f"{intro}\n\n{bullets[0]}", ["shap", "sign", "positive"]),
                ("signo-negativo", f"{intro}\n\n{bullets[1]}", ["shap", "sign", "negative"]),
                ("no-causalidad", f"{bullets[2]}\n\n{causal}", ["shap", "non-causality"]),
            ]
            for name, unit, tags in units:
                chunks.append(_make_chunk(source, "shap_guidelines", metadata, title,
                                          section, name, f"## {section}\n\n{unit}", tags))
        else:
            tags = ["shap"] + (["limitations"] if section == "Limitaciones" else [])
            chunks.append(_make_chunk(source, "shap_guidelines", metadata, title,
                                      section, "", f"## {section}\n\n{content}", tags))
    return chunks


def chunk_document(path: Path) -> list[dict[str, object]]:
    metadata, body = _front_matter(path.read_text(encoding="utf-8"))
    if metadata["kb_version"] != 1 or metadata["model_version"] != "v2" or metadata["status"] != "current":
        raise ValueError(f"Excluded non-current knowledge-base document: {path.name}")
    title, sections = _sections(body)
    source = path.name
    doc_type = str(metadata.get("scope", path.stem))
    if source == "data_dictionary.md":
        return _data_dictionary_chunks(source, metadata, title, sections)
    if source == "modeling_methodology.md":
        return _methodology_chunks(source, metadata, title, sections)
    if source == "shap_interpretation_guidelines.md":
        return _shap_chunks(source, metadata, title, sections)
    if source == "provenance_and_versioning.md":
        # The first block of this section enumerates content explicitly excluded
        # from retrieval (including an obsolete LLM).  Retain only the current
        # operational statement; excluded history must not enter the corpus.
        sections = [
            (section, _paragraphs(content)[-1] if section == "Exclusiones operativas" else content)
            for section, content in sections
        ]
    return [
        _make_chunk(source, doc_type, metadata, title, section, "",
                    f"## {section}\n\n{content}", [])
        for section, content in sections
    ]


def build_chunks(kb_dir: Path) -> list[dict[str, object]]:
    paths = sorted(kb_dir.glob("*.md"), key=lambda path: path.name)
    if len(paths) != 9:
        raise ValueError(f"Expected exactly 9 knowledge-base documents, found {len(paths)}")
    chunks = [chunk for path in paths for chunk in chunk_document(path)]
    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs are not unique")
    if any(not str(chunk["text"]).strip() for chunk in chunks):
        raise ValueError("Empty chunk generated")
    return chunks


def resolve_expected_chunk_ids(record: dict[str, object], chunks: list[dict[str, object]]) -> list[str]:
    """Resolve only chunks supported by an expected source/section pair.

    Broad feature-table references are narrowed to technical feature names found
    in the immutable evaluation fields.  Split semantic sections are narrowed by
    question/facts vocabulary; otherwise all exact section matches are retained.
    """
    sources = set(record.get("expected_sources", []))
    sections = [str(value).strip("`") for value in record.get("expected_sections", [])]
    evidence = " ".join(
        str(record.get(field, ""))
        for field in ("question", "expected_answer", "expected_facts", "model_context")
    ).lower()
    candidates = [
        chunk for chunk in chunks
        if chunk["source"] in sources
        and any(f" > {section}" in str(chunk["section_path"]) for section in sections)
    ]
    resolved: list[dict[str, object]] = []
    for chunk in candidates:
        feature = chunk.get("feature_name")
        if feature and str(feature).lower() not in evidence:
            continue
        chunk_id = str(chunk["chunk_id"])
        if "--alcance--" in chunk_id:
            if "positivo" in evidence or "eleva" in evidence:
                wanted = ("signo-positivo", "no-causalidad")
            elif "negativ" in evidence or "reduce" in evidence:
                wanted = ("signo-negativo", "no-causalidad")
            else:
                wanted = ("interpretacion", "no-causalidad")
            if not any(chunk_id.endswith(item) for item in wanted):
                continue
        if "--oof-y-seleccion-de-threshold--" in chunk_id:
            if "youden" in evidence or "threshold" in evidence:
                wanted = ("youden", "threshold")
            else:
                wanted = ("oof",)
            if not any(chunk_id.endswith(item) for item in wanted):
                continue
        resolved.append(chunk)
    return sorted(str(chunk["chunk_id"]) for chunk in resolved)
