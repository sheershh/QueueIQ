"""Screenshots of the dashboard for review and the README (`make screenshots`).

Needs the app running (default http://localhost:8501; pass another URL as the first argument)
and Google Chrome (Playwright drives it; no browser download). WebGL for the 3D map runs on
SwiftShader in headless mode. Writes docs/screenshots/*.png at 1440 px and 390 px wide.
"""

import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
SIZES = {"laptop": (1440, 3600), "phone": (390, 7000)}
IDLE = (
    "document.querySelector('[data-testid=\"stApp\"]')"
    "?.getAttribute('data-test-script-state') === 'notRunning'"
)


def settle(page: Page, ms: int = 1500) -> None:
    """Wait until Streamlit has finished rerunning."""
    page.wait_for_timeout(300)
    page.wait_for_function(IDLE, timeout=90_000, polling=200)
    page.wait_for_timeout(ms)


def shoot(page: Page, name: str) -> None:
    path = OUT / f"{name}.png"
    box = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
    foot = page.locator(".qi-footer").bounding_box()
    if foot:  # stop just below the footer: the container can be taller than its content
        box["height"] = foot["y"] + foot["height"] + 32 - box["y"]
    page.screenshot(path=str(path), clip=box)
    # 256-color palette: these flat-color pages look the same and stay well under 1 MB.
    img = Image.open(path).convert("RGB")
    img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        path, optimize=True
    )
    print(f"Wrote {path.relative_to(OUT.parents[1])}")


def click_a_county(page: Page) -> str:
    """Click points on the map until a county is selected; returns the selected label."""
    label = page.locator(".qi-pctl-value").first
    before = label.inner_text()
    box = page.locator('[data-testid="stDeckGlJsonChart"]').bounding_box()
    # Headless WebGL (SwiftShader) picking misses some clicks that work in a real browser, so try
    # a grid of points until one selects a county.
    points = [(fx, fy) for fx in (0.6, 0.7) for fy in (0.55, 0.45)]
    for fx, fy in points:
        page.mouse.click(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
        settle(page, 1500)
        if label.inner_text() != before:
            return label.inner_text()
    # Fall back to the keyboard-accessible list, which drives the same selection and fly-to.
    page.get_by_text("Choose from a list instead").click()
    settle(page, 500)
    page.locator('input[aria-label*="County"]').first.click()
    page.keyboard.type("Washington")
    page.keyboard.press("Enter")
    settle(page, 3000)
    return label.inner_text() + " (via the list)"


def open_tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    settle(page, 2000)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome", args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        )
        for size, (w, h) in SIZES.items():
            # Phone pages are ~4,500 px tall: at 2x they pass Chrome's 8,192-pixel limit and tile.
            scale = 2 if size == "laptop" else 1.5
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=scale)
            page.goto(URL)
            page.wait_for_selector('[data-testid="stDeckGlJsonChart"]', timeout=120_000)
            settle(page, 6000)
            shoot(page, f"overview_{size}")
            if size == "laptop":
                print("selected:", click_a_county(page))
                shoot(page, "county_selected_laptop")
                open_tab(page, "Why")
                shoot(page, "why_laptop")
                open_tab(page, "Shortlist")
                for tech in ("Solar", "Gas", "Battery"):
                    page.locator('[data-testid="stButtonGroup"] button').filter(
                        has_text=tech
                    ).first.click()
                    settle(page, 800)
                    page.locator('input[aria-label="Name for this setup"]').fill(f"{tech} 150 MW")
                    page.locator('input[aria-label="Name for this setup"]').press("Enter")
                    settle(page, 500)
                    page.get_by_role("button", name="Add to shortlist").click()
                    settle(page, 1500)
                shoot(page, "shortlist_laptop")
                page.locator('input[aria-label="Capacity (MW)"]').fill("0")
                page.locator('input[aria-label="Capacity (MW)"]').press("Enter")
                settle(page, 1000)
                shoot(page, "error_laptop")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
