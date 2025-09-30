#!/usr/bin/env python3
"""Download daily/seasonal insolation forcings from NOAA NCEI.

The script mirrors the public directory under
``https://www.ncei.noaa.gov/pub/data/paleo/climate_forcing/orbital_variations/insolation/``
into ``data/raw/insolation/`` relative to the project root. Existing files are
left untouched so the download can be resumed without re-transferring data.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

BASE_URL = (
    "https://www.ncei.noaa.gov/pub/data/paleo/"
    "climate_forcing/orbital_variations/insolation/"
)
USER_AGENT = "Mozilla/5.0 (compatible; green-wave-downloader/1.0)"


class DirectoryListingParser(HTMLParser):
    """Minimal HTML parser to extract file links from Apache-style listings."""

    def __init__(self) -> None:
        super().__init__()
        self.files: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_dict = dict(attrs)
        href = attr_dict.get("href")
        if not href:
            return
        if href.startswith("?") or href.startswith("#"):
            return
        if href in {"/", "../"}:
            return
        if href.endswith("/"):
            # Ignore sub-directories; the dataset is flat.
            return
        self.files.append(href)


def fetch_directory_listing(url: str) -> Iterable[str]:
    """Return iterable of filenames exposed at *url*.

    The NOAA directory serves a simple HTML listing. We request the page with a
    browser-like user-agent to avoid being rejected by their CDN and then parse
    the anchor tags to recover file names.
    """

    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request) as response:
            content_type = response.headers.get("Content-Type", "")
            encoding = "utf-8"
            if "charset=" in content_type:
                encoding = content_type.split("charset=")[-1]
            html = response.read().decode(encoding, errors="replace")
    except HTTPError as exc:  # pragma: no cover - network specific
        raise RuntimeError(f"Failed to list remote directory: {exc}") from exc
    except URLError as exc:  # pragma: no cover - network specific
        raise RuntimeError(f"Failed to reach remote directory: {exc}") from exc

    parser = DirectoryListingParser()
    parser.feed(html)
    return parser.files


@dataclass
class DownloadResult:
    filename: str
    skipped: bool
    reason: str | None = None


def download_file(base_url: str, filename: str, destination: Path, overwrite: bool) -> DownloadResult:
    """Download *filename* relative to *base_url* into *destination*."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        return DownloadResult(filename=filename, skipped=True, reason="exists")

    file_url = urljoin(base_url, filename)
    request = Request(file_url, headers={"User-Agent": USER_AGENT})

    try:
        with urlopen(request) as response, destination.open("wb") as output:
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
    except HTTPError as exc:  # pragma: no cover - network specific
        raise RuntimeError(f"Failed to download {filename}: {exc}") from exc
    except URLError as exc:  # pragma: no cover - network specific
        raise RuntimeError(f"Failed to reach {file_url}: {exc}") from exc

    return DownloadResult(filename=filename, skipped=False)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Remote directory with the insolation files.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/insolation"),
        help="Local directory where the files will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that *would* be downloaded without transferring anything.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=True)

    try:
        filenames = list(fetch_directory_listing(args.base_url))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not filenames:
        print("No files discovered at the remote directory.")
        return 0

    print(f"Found {len(filenames)} files at {args.base_url}")

    if args.dry_run:
        for name in filenames:
            local = destination / name
            status = "exists" if local.exists() else "missing"
            print(f"{name} -> {local} ({status})")
        return 0

    for name in filenames:
        target = destination / name
        try:
            result = download_file(args.base_url, name, target, overwrite=args.force)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

        if result.skipped:
            print(f"Skipping {name} (already exists)")
        else:
            print(f"Downloaded {name}")

    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
