#!/usr/bin/env python3
"""Fetch a YouTube transcript and write it to data/raw as an anchored .md.

Stage ① Acquire of the pipeline: pull captions once,
offline, and commit the result. Nothing downstream re-fetches.

Output carries YAML front-matter -- the identity a record's provenance hangs
off, written into the file rather than implied by its name -- then a body of
captions bucketed into fixed-second intervals, one anchor per line.

    python -m carryia.pipeline.fetch_transcript URL --creator-id skill-capped
    python -m carryia.pipeline.fetch_transcript URL --creator-id skill-capped \\
        --source-id skill-capped-complete-beginner-guide-2026 --interval 15
    python -m carryia.pipeline.fetch_transcript VIDEOID --creator-id x --interval 0  # per-cue
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    YouTubeTranscriptApi,
)

from carryia.paths import DATA

DEFAULT_OUT_DIR = DATA / "raw"
DEFAULT_INTERVAL = 10
OEMBED = "https://www.youtube.com/oembed"
WATCH = "https://www.youtube.com/watch?v={}"

VIDEO_ID = re.compile(r"^[0-9A-Za-z_-]{11}$")
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
    "www.youtu.be",
}
# Path shapes that carry the id as the segment after the prefix.
PATH_PREFIXES = ("shorts", "embed", "live", "v", "e")


class TranscriptError(Exception):
    """Anything that should surface as a one-line message, not a traceback."""


def parse_video_id(url: str) -> str:
    """Accept a watch/short/embed/live URL, a youtu.be link, or a bare id."""
    # Backslashes are never valid here; a quoted URL from a shell with
    # url-quote-magic arrives with \? and \= still in it.
    url = url.strip().replace("\\", "")
    if VIDEO_ID.match(url):
        return url

    parsed = urlparse(url if "//" in url else f"https://{url}")
    host = parsed.netloc.lower()
    if host not in YOUTUBE_HOSTS:
        raise TranscriptError(f"not a YouTube URL: {url}")

    segments = [s for s in parsed.path.split("/") if s]
    candidate = ""
    if host.endswith("youtu.be"):
        candidate = segments[0] if segments else ""
    elif segments and segments[0] == "watch":
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    elif len(segments) >= 2 and segments[0] in PATH_PREFIXES:
        candidate = segments[1]

    if not VIDEO_ID.match(candidate):
        raise TranscriptError(f"could not find a video id in: {url}")
    return candidate


def fetch_snippets(video_id: str, languages: list[str]) -> tuple[list, str]:
    """Return (snippets, source_label), preferring human captions over auto ones.

    Auto-generated captions have no punctuation and mangle proper nouns, which
    is what the cleanup stage exists to repair -- but a manual track skips that
    cost entirely, so always try for one first.
    """
    api = YouTubeTranscriptApi()
    try:
        available = api.list(video_id)
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptError(f"{video_id}: {exc.cause or exc}") from exc

    for label, finder in (
        ("manual", available.find_manually_created_transcript),
        ("auto-generated", available.find_generated_transcript),
    ):
        try:
            transcript = finder(languages)
        except NoTranscriptFound:
            continue
        return list(transcript.fetch()), f"{transcript.language_code} ({label})"

    offered = ", ".join(sorted(t.language_code for t in available)) or "none"
    raise TranscriptError(
        f"{video_id}: no transcript in {languages}; available: {offered}"
    )


def fetch_title(video_id: str) -> str:
    """Look the title up via oEmbed so it lands in the file verbatim.

    Only a convenience for --title; a wrong or missing title is a provenance
    hole, so failure here asks for the flag rather than guessing.
    """
    query = urlencode({"url": WATCH.format(video_id), "format": "json"})
    try:
        response = requests.get(f"{OEMBED}?{query}", timeout=10)
        response.raise_for_status()
        title = response.json()["title"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise TranscriptError(
            f"{video_id}: could not look up the title ({exc}); pass --title"
        ) from exc
    return " ".join(title.split())


def bucket(snippets: list, interval: int) -> list[tuple[int, str]]:
    """Group cues into fixed-second windows, floored to the interval boundary.

    interval=0 keeps one line per cue, anchored at its own start second.
    """
    lines: list[tuple[int, str]] = []
    for snippet in snippets:
        text = " ".join(snippet.text.split())
        if not text:
            continue
        second = int(snippet.start)
        if interval:
            second -= second % interval
        if lines and lines[-1][0] == second:
            lines[-1] = (second, f"{lines[-1][1]} {text}")
        else:
            lines.append((second, text))
    return lines


def render(
    lines: list[tuple[int, str]],
    video_id: str,
    args: argparse.Namespace,
    captions: str,
) -> str:
    """Front-matter carrying identity + provenance, then anchored body lines.

    creator_id is the corroboration axis -- the "no creator above 60%" gate in
    validation counts on it, and two videos from the same creator are only
    knowably the same opinion because they share it.
    """
    front_matter = {
        "source_id": args.source_id,
        "creator_id": args.creator_id,
        "medium": "video",
        "title": args.title,
        "source_url": WATCH.format(video_id),
        "retrieved_at": date.today().isoformat(),
        "captions": captions,
        "interval": f"{args.interval}s" if args.interval else "per-cue",
    }
    # JSON strings are valid YAML double-quoted scalars, so this escapes any
    # colon, quote or pipe a video title happens to contain.
    header = ["---"]
    header += [f"{key}: {json.dumps(value)}" for key, value in front_matter.items()]
    header += ["---", ""]

    body = [f"[{second}] {text}" for second, text in lines]
    return "\n".join(header + body) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0]) # type: ignore
    parser.add_argument("url", help="YouTube URL or bare 11-character video id")
    parser.add_argument(
        "--creator-id",
        required=True,
        metavar="SLUG",
        help="the opinion behind the source, e.g. skill-capped; shared across "
        "everything that creator publishes (required -- validation counts on it)",
    )
    parser.add_argument(
        "--source-id",
        metavar="SLUG",
        help="identity of this document, and the output stem "
        "(default: the video id)",
    )
    parser.add_argument(
        "--title",
        help="source title; looked up from YouTube when omitted",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"seconds per line; 0 = one line per caption cue (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["en"],
        metavar="CODE",
        help="preferred language codes, best first (default: en)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing file",
    )

    args = parser.parse_args(argv)
    if args.interval < 0:
        parser.error("--interval must be 0 or greater")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        video_id = parse_video_id(args.url)
        args.source_id = args.source_id or video_id
        out_path = args.out_dir / f"{args.source_id}.md"
        if out_path.exists() and not args.force:
            raise TranscriptError(f"{out_path} already exists (use --force to replace)")

        args.title = args.title or fetch_title(video_id)
        snippets, captions = fetch_snippets(video_id, args.lang)
        lines = bucket(snippets, args.interval)
        if not lines:
            raise TranscriptError(f"{video_id}: transcript is empty")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render(lines, video_id, args, captions), encoding="utf-8")
    except TranscriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    last = lines[-1][0]
    print(f"{out_path}  ({len(lines)} lines, {captions}, ends {last // 60}m{last % 60:02d}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
