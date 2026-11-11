"""Screenshots of the dashboard for review and the README (`make screenshots`).

Needs the app running on http://localhost:8501 (`make app` or `docker compose up`) and Google
Chrome installed (Playwright drives it; no browser download). Writes docs/screenshots/*.png at a
laptop width (1440 px) and a phone width (390 px).
"""

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
SIZES = {"laptop": (1440, 1400), "phone": (390, 2600)}


def settle(page: Page, ms: int = 1500) -> None:
    """Wait until Streamlit has finished rerunning: no 'Running' widget, no stale elements."""
    page.wait_for_timeout(300)
    page.wait_for_function(
        "document.querySelector('[data-testid=\"stApp\"]')"
        "?.getAttribute('data-test-script-state') === 'notRunning'",
        timeout=60_000,
        polling=200,
    )
    page.wait_for_timeout(ms)


def choose(page: Page, label: str, option: str) -> None:
    page.locator(f'input[aria-label*="{label}"]').first.click()
    page.get_by_role("option", name=option, exact=True).click()
    settle(page, 800)


def add(page: Page, name: str) -> None:
    box = page.get_by_label("Name", exact=True).and_(page.locator("input"))
    box.fill(name)
    box.press("Enter")
    settle(page, 500)
    page.get_by_role("tab", name="Compare").click()
    page.get_by_role("button", name="Add current project to comparison").click()
    page.get_by_text(f"{name}:", exact=False).first.wait_for(timeout=60_000)
    settle(page, 800)


def shoot(page: Page, name: str) -> None:
    path = OUT / f"{name}.png"
    page.locator('[data-testid="stMainBlockContainer"]').screenshot(path=str(path))
    print(f"Wrote {path.relative_to(OUT.parents[1])}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        for size, (w, h) in SIZES.items():
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            page.goto(URL)
            page.wait_for_selector('[data-testid="stVegaLiteChart"]', timeout=60_000)
            settle(page, 2500)
            shoot(page, f"outlook_{size}")

            add(page, "Solar 150 MW")
            choose(page, "Technology", "Gas")
            add(page, "Gas 150 MW")
            choose(page, "Technology", "Battery storage")
            add(page, "Battery 150 MW")
            page.get_by_role("tab", name="Compare").click()
            settle(page, 2000)
            shoot(page, f"compare_{size}")

            page.get_by_role("tab", name="Why this prediction").click()
            settle(page, 2500)
            shoot(page, f"why_{size}")

            choose(page, "Grid operator", "CAISO")
            page.locator('input[aria-label="Capacity (MW)"]').fill("0")
            page.locator('input[aria-label="Capacity (MW)"]').press("Enter")
            settle(page, 800)
            page.get_by_role("tab", name="Outlook").click()
            settle(page, 800)
            shoot(page, f"error_{size}")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
