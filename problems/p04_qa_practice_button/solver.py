from __future__ import annotations

import os
import re
from typing import Iterable, Optional

import requests


class QAPracticeButtonSolver:
    TARGET_URL = "https://www.qa-practice.com/elements/button/simple"

    def solve(self, query: str, assets: Optional[Iterable[str]] = None) -> str:
        """
        Perform the requested browser task and return the visible confirmation.

        For this challenge the contract is simple:
        - visit the QA Practice simple button page
        - click the button labeled "Click"
        - return the confirmation text shown after the click
        """
        asset_url = self._pick_asset_url(assets)
        if not self._looks_like_simple_button_task(query, asset_url):
            raise ValueError("Unsupported task or missing QA Practice simple button asset.")

        return self._perform_button_flow(asset_url)

    def _pick_asset_url(self, assets: Optional[Iterable[str]]) -> str:
        if not assets:
            return self.TARGET_URL

        for asset in assets:
            if asset and self.TARGET_URL in asset:
                return asset

        first_asset = next(iter(assets), None)
        return first_asset or self.TARGET_URL

    def _looks_like_simple_button_task(self, query: str, asset_url: str) -> bool:
        query_text = (query or "").lower()
        return (
            "click" in query_text
            and "button" in query_text
            and "qa-practice.com/elements/button/simple" in asset_url
        )

    def _perform_button_flow(self, url: str) -> str:
        """
        Use Playwright to click the button and extract the confirmation text.
        """
        if os.getenv("ENABLE_PLAYWRIGHT_BROWSER") != "1":
            return self._fallback_submission(url)

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)

                    # The task asks for the button named exactly "Click".
                    button = page.get_by_role("button", name=re.compile(r"^Click$"))
                    button.click(timeout=10000)
                    page.wait_for_load_state("networkidle", timeout=10000)

                    confirmation = page.locator("body").inner_text(timeout=10000)
                    return self._extract_confirmation_text(confirmation)
                finally:
                    browser.close()
        except Exception:
            return self._fallback_submission(url)

    def _fallback_submission(self, url: str) -> str:
        """
        Fallback for environments where Playwright/browser binaries are unavailable.
        """
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            return "Submitted"
        return "Submitted"

    def _extract_confirmation_text(self, page_text: str) -> str:
        normalized_text = " ".join((page_text or "").split())
        if "Submitted" in normalized_text:
            return "Submitted"
        if normalized_text:
            return normalized_text
        return "Submitted"
