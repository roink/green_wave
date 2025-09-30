"""Download insolation forcing files from the NOAA Paleoclimatology archive.

This script mirrors the public directory that stores daily/seasonal
insolation reconstructions from Berger-style orbital solutions.  Files are
retrieved from
https://www.ncei.noaa.gov/pub/data/paleo/climate_forcing/orbital_variations/insolation/
into the project-local ``data/raw/insolation`` directory.  Only files that are
missing locally are downloaded so that the script can be run repeatedly without
wasting bandwidth.
"""

from __future__ import annotations

import argparse
import logging
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import shutil

# Default configuration for the orbital forcing archive.
DEFAULT_BASE_URL = (
    "https://www.ncei.noaa.gov/pub/data/paleo/climate_forcing/"
    "orbital_variations/insolation/"
)
DEFAULT_OUTPUT_DIR = Path("data/raw/insolation")
USER_AGENT = "green-wave-data-fetch/1.0 (+https://www.noaa.gov/)"


class DirectoryListingParser(HTMLParser):
    """Simple HTML parser that extracts ``href`` entries from anchor tags."""

    def __init__(self) -> None:
        super().__init__()
        self._links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str]]) -> None:  # type: ignore[override]
        if tag.lower() != "a":
            return
        attr_dict = dict(attrs)
        href = attr_dict.get("href")
        if href:
            self._links.append(href)

    @property
    def links(self) -> List[str]:
        return self._links


def list_remote_files(base_url: str) -> List[str]:
    """Return a list of file names available at ``base_url``.

    Parameters
    ----------
    base_url: str
        Remote directory to scan for files.
    """

    request = Request(base_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:  # nosec: B310 - URL controlled by script input
        content = response.read().decode("utf-8", errors="ignore")

    parser = DirectoryListingParser()
    parser.feed(content)

    files = []
    for href in parser.links:
        if href in {"../", "?C=N;O=D", "?C=M;O=A", "?C=S;O=A", "?C=D;O=A"}:
            continue
        if href.endswith("/"):
            # Ignore sub-directories; the NOAA directory does not contain nested
            # folders for the insolation tables.
            continue
        files.append(href)
    return sorted(set(files))


def download_file(base_url: str, filename: str, output_dir: Path) -> bool:
    """Download a single file if it is not already present.

    Returns ``True`` when a download occurred and ``False`` if the file was
    already available locally.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename

    if destination.exists():
        return False

    file_url = urljoin(base_url, filename)
    request = Request(file_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:  # nosec: B310 - trusted NOAA domain
        with open(destination, "wb") as fh:
            shutil.copyfileobj(response, fh)

    return True


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(message)s", level=level)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download insolation forcing tables from the NOAA "
            "Paleoclimatology archive."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Remote directory that contains the insolation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to store the downloaded files.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Show which files would be downloaded without retrieving them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information during the download process.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    logger = logging.getLogger("download_insolation")

    try:
        remote_files = list_remote_files(args.base_url)
    except (HTTPError, URLError) as exc:
        logger.error("Failed to retrieve directory listing: %s", exc)
        return 1

    if not remote_files:
        logger.warning("No files found at %s", args.base_url)
        return 1

    logger.info("Found %d files in remote directory", len(remote_files))

    if args.list_only:
        for name in remote_files:
            logger.info(name)
        return 0

    downloaded = 0
    skipped = 0
    for filename in remote_files:
        try:
            did_download = download_file(args.base_url, filename, args.output_dir)
        except (HTTPError, URLError) as exc:
            logger.error("Failed to download %s: %s", filename, exc)
            continue

        if did_download:
            downloaded += 1
            logger.info("Downloaded %s", filename)
        else:
            skipped += 1
            logger.debug("Already present: %s", filename)

    logger.info(
        "Completed. %d new file(s) downloaded, %d skipped.", downloaded, skipped
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
