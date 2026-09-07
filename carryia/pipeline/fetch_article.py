#!/usr/bin/env python3
"""Structure a saved HTML guide into an anchored raw .md (stage 1 for written sources).

The written-source twin of carryia/pipeline/fetch_transcript.py. A video hands you a
timeline, so that script buckets by timestamp; an article hands you a section
structure, so this one splits by heading -- but every site marks its sections
differently, so the per-site logic lives in a small ADAPTERS registry and the
engine around it stays generic.

Input is a LOCAL html file you saved yourself (browser "Save As -> Webpage,
Complete", or DevTools "Copy outerHTML"). Saving by hand is deliberate: it
dodges Cloudflare bot-gates and captures JS-rendered content a scripted GET
would miss, and the committed file becomes the provenance ground truth.

    python -m carryia.pipeline.fetch_article page.html --creator-id eiensiei
    python -m carryia.pipeline.fetch_article page.html --creator-id mobalytics \\
        --source-id wave-management --adapter mobalytics.gg

Output mirrors fetch_transcript.py: YAML front-matter (medium: "written") then a
body of `## Heading` sections, each an anchor a distilled tip's `section` cites.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString

from carryia.paths import DATA

DEFAULT_OUT_DIR = DATA / "raw"
PARSER = "html.parser"
# Headings that are pure navigation, dropped whatever the source.
CHROME_HEADINGS = {"table of contents"}
# Below this many characters a "section" is a stub (a divider, an empty block),
# not coaching prose -- skipped so it can't dilute the ## anchor set.
MIN_BODY_CHARS = 40


class ArticleError(Exception):
    """Anything that should surface as a one-line message, not a traceback."""


def collapse(text: str) -> str:
    """One line, single-spaced. Entities are already decoded by BeautifulSoup.

    str.split() folds every run of whitespace -- including the NBSP (\\xa0) HTML
    loves -- so the body reads clean and offsets into it stay stable.
    """
    return " ".join(text.split())


# --- per-site adapters: soup -> [(title, body_text)] in document order --------

def mobalytics(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Semantic site: one <h2> per section, body is the text until the next <h2>.

    Its class names are hashed (StyleX: x1n2onr6...) and rotate every deploy, so
    nothing here keys on a class -- only on the <h2> tag, which is stable. The
    heading elements are swapped for sentinels so a single get_text() flatten
    can be split back into (title, body) pairs regardless of how they nest.
    """
    root = soup.find("article") or soup.find("main") or soup.body or soup
    headings = root.find_all("h2")
    if not headings:
        return []
    marker = "\ue000SECTION\ue000"
    for h in headings:
        title = h.get_text(" ", strip=True)
        h.replace_with(NavigableString(f"{marker}{title}{marker}"))
    chunks = root.get_text("\n").split(marker)
    # chunks == [preamble, title1, body1, title2, body2, ...]
    sections = []
    for i in range(1, len(chunks), 2):
        title = collapse(chunks[i])
        body = chunks[i + 1] if i + 1 < len(chunks) else ""
        sections.append((title, body))
    return sections


def mobafire(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Community guides: chapters are <div class="view-guide__chapter"> (stable
    BEM classes), but the title isn't inside the block -- it lives in the side
    table-of-contents. Zip the ToC titles onto the chapter bodies in order.
    """
    chapters = soup.select(".view-guide__chapters .view-guide__chapter")
    titles = [a.get_text(" ", strip=True)
              for a in soup.select(".side-toc a[href*='#chapter']")]
    sections = []
    for i, chapter in enumerate(chapters):
        content = chapter.select_one(".view-guide__chapter__content")
        if content is None:
            continue
        anchor = chapter.find(attrs={"name": True})
        title = (titles[i] if i < len(titles)
                 else anchor["name"] if anchor else f"section-{i + 1}")
        sections.append((title, content.get_text("\n")))
    return sections


ADAPTERS = {"mobalytics.gg": mobalytics, "mobafire.com": mobafire}


# --- generic engine -----------------------------------------------------------

def canonical_url(soup: BeautifulSoup) -> str | None:
    link = soup.find("link", attrs={"rel": "canonical"})
    if link and link.get("href"):
        return link["href"]
    og = soup.find("meta", attrs={"property": "og:url"})
    return og["content"] if og and og.get("content") else None


def page_title(soup: BeautifulSoup) -> str | None:
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return collapse(og["content"])
    return collapse(soup.title.get_text()) if soup.title else None


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def pick_adapter(host: str, override: str | None):
    key = override or host
    if key not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS)) or "none"
        raise ArticleError(
            f"no adapter for {key!r}; known: {known} (pass --adapter, or add one)"
        )
    return key, ADAPTERS[key]


def sectionize(raw_sections, source_text):
    """Drop chrome/stub sections, then prove each body was faithfully extracted.

    The verbatim guard: a collapsed section body must be a literal substring of
    the collapsed page text. BeautifulSoup only ever *selects* text, so this
    holds unless an adapter reorders or fabricates -- exactly the bug worth
    failing on, since source_excerpt later slices offsets into this very file.
    """
    kept = []
    for title, raw_body in raw_sections:
        body = collapse(raw_body)
        if title.lower() in CHROME_HEADINGS or len(body) < MIN_BODY_CHARS:
            continue
        if body not in source_text:
            raise ArticleError(
                f"section {title!r} is not a literal substring of the source "
                "-- the adapter altered text; fix the adapter, don't paper over it"
            )
        kept.append((title, body))
    if not kept:
        raise ArticleError("no sections survived extraction -- wrong adapter?")
    return kept


def render(sections, meta) -> str:
    """Front-matter carrying identity + provenance, then `## Heading` sections.

    `adapter` is the written analogue of the video file's `captions`/`interval`:
    it records *which* site rules produced this text, so a reviewer can trace it.
    """
    header = ["---"]
    header += [f"{key}: {json.dumps(value)}" for key, value in meta.items()]
    header += ["---", ""]
    body = []
    for title, text in sections:
        body += [f"## {title}", "", text, ""]
    return "\n".join(header + body).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # type: ignore
    parser.add_argument("html", type=Path, help="local saved .html file")
    parser.add_argument(
        "--creator-id",
        required=True,
        metavar="SLUG",
        help="the opinion/author behind the source (validation's 'no creator "
        "above 60%%' gate counts on it -- use the author, not the platform)",
    )
    parser.add_argument(
        "--source-id",
        metavar="SLUG",
        help="document identity + output stem "
        "(default: last path segment of the canonical URL)",
    )
    parser.add_argument(
        "--adapter",
        metavar="HOST",
        help=f"force a site adapter; known: {', '.join(sorted(ADAPTERS))}",
    )
    parser.add_argument("--title", help="source title (default: og:title / <title>)")
    parser.add_argument("--source-url", help="citation URL (default: canonical / og:url)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not args.html.is_file():
            raise ArticleError(f"no such file: {args.html}")
        soup = BeautifulSoup(args.html.read_text(encoding="utf-8", errors="replace"), PARSER)
        for junk in soup(["script", "style", "noscript", "template"]):
            junk.decompose()

        source_url = args.source_url or canonical_url(soup)
        if not source_url:
            raise ArticleError("no canonical/og:url in the page; pass --source-url")
        title = args.title or page_title(soup)
        if not title:
            raise ArticleError("no og:title/<title> in the page; pass --title")

        adapter_name, adapter = pick_adapter(host_of(source_url), args.adapter)
        # Capture the ground-truth text before any adapter mutates the tree.
        source_text = collapse(soup.get_text(" "))
        sections = sectionize(adapter(soup), source_text)

        source_id = args.source_id or urlparse(source_url).path.rstrip("/").split("/")[-1]
        if not source_id:
            raise ArticleError("could not derive a source-id from the URL; pass --source-id")
        out_path = args.out_dir / f"{source_id}.md"
        if out_path.exists() and not args.force:
            raise ArticleError(f"{out_path} already exists (use --force to replace)")

        meta = {
            "source_id": source_id,
            "creator_id": args.creator_id,
            "medium": "written",
            "title": title,
            "source_url": source_url,
            "retrieved_at": date.today().isoformat(),
            "adapter": adapter_name,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render(sections, meta), encoding="utf-8")
    except ArticleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    chars = sum(len(text) for _, text in sections)
    print(f"{out_path}  ({len(sections)} sections, {chars:,} chars, adapter={adapter_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
