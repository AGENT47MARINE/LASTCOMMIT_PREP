"""
Wikipedia infobox image scraper.

Dependencies:
    pip install requests beautifulsoup4
"""

from __future__ import annotations

import sys

import requests
from bs4 import BeautifulSoup


def extract_infobox_image_src(wikipedia_url: str) -> str | None:
    """
    Return the best-available image URL from a Wikipedia page, or None if not found.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(wikipedia_url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error fetching page: {exc}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Prefer the sports sidebar image when available (e.g., Olympic rings icon).
    sidebar = soup.find(
        "table",
        class_=lambda c: c and "sidebar" in c and "multi-sport-event-sidebar" in c,
    )

    img_tag = sidebar.find("img") if sidebar else None

    # Fallback to the first image inside an infobox table.
    if not img_tag:
        infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
        if infobox:
            img_tag = infobox.find("img")

    # Final fallback: first image in article content for pages without infobox/sidebar.
    if not img_tag:
        img_tag = soup.select_one("#mw-content-text img")

    if not img_tag:
        print("No image found on this page")
        return None

    src = img_tag.get("src")
    if not src:
        print("No image found on this page")
        return None

    print(src)
    return src


def main() -> None:
    """
    Accept a URL from command-line argument or interactive input, then scrape.
    """
    if len(sys.argv) > 1:
        wikipedia_url = sys.argv[1].strip()
    else:
        wikipedia_url = input("Enter Wikipedia URL: ").strip()

    if not wikipedia_url:
        print("Please provide a Wikipedia URL")
        return

    extract_infobox_image_src(wikipedia_url)


if __name__ == "__main__":
    main()
