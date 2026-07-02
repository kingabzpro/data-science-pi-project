"""Webwright final script: test the Adult Income Streamlit app in Chromium."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

RUN_DIR = Path(__file__).parent
SCREENSHOTS = RUN_DIR / "screenshots"
LOG = RUN_DIR / "final_script_log.txt"
PROJECT_ROOT = RUN_DIR.parents[3]
APP_PORT = 8501
APP_URL = f"http://localhost:{APP_PORT}"


def log(step: int, msg: str) -> None:
    line = f"step {step} action: {msg}\n"
    LOG.open("a", encoding="utf-8").write(line)
    print(line, end="")


def start_streamlit() -> subprocess.Popen:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv not found on PATH")
    cmd = [
        uv,
        "run",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.port",
        str(APP_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_server(url: str, timeout: float = 60.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def fill_number(page, label: str, value: str) -> None:
    """Clear and fill a Streamlit number input by its label."""
    field = page.get_by_role("spinbutton", name=label)
    field.click()
    field.fill(value)


def select_option(page, label: str, value: str) -> None:
    """Open a Streamlit selectbox, type to filter, and click the option.

    Streamlit selectboxes render as a custom combobox. Long option lists are
    virtualised, so typing a prefix filters the list and guarantees the
    target option is present in the DOM before we click it.
    """
    combo = page.locator(f'input[role="combobox"][aria-label*="{label}"]')
    combo.click()
    # Type a prefix to filter long lists (handles virtualised dropdowns).
    prefix = value.split("-")[0] if "-" in value else value[:4]
    combo.type(prefix)
    page.wait_for_timeout(300)
    option = page.get_by_role("option", name=value, exact=True)
    option.wait_for(state="visible", timeout=10000)
    option.click()
    page.wait_for_timeout(300)


def main() -> int:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")

    console_errors: list[dict] = []

    proc = start_streamlit()
    try:
        log(0, f"started streamlit subprocess (pid={proc.pid})")
        if not wait_for_server(APP_URL):
            log(1, "ERROR: streamlit server did not become ready")
            return 1
        log(1, "streamlit server is ready")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 1800})
            page = context.new_page()

            page.on(
                "console",
                lambda msg: console_errors.append(
                    {"type": msg.type, "text": msg.text, "location": msg.location}
                ),
            )
            page.on(
                "pageerror",
                lambda exc: console_errors.append(
                    {"type": "pageerror", "text": str(exc), "location": {}}
                ),
            )

            page.goto(APP_URL, wait_until="networkidle")
            page.screenshot(path=str(SCREENSHOTS / "final_execution_2_homepage.png"))
            log(2, "loaded streamlit app homepage")

            # CP2: verify form controls are visible
            page.get_by_role("spinbutton", name="Age").wait_for(state="visible")
            page.get_by_role("spinbutton", name="Years of education").wait_for(
                state="visible"
            )
            page.get_by_role("combobox", name="Workclass").wait_for(state="visible")
            page.screenshot(
                path=str(SCREENSHOTS / "final_execution_3_form_visible.png")
            )
            log(3, "prediction form controls are visible")

            # Fill numeric inputs
            fill_number(page, "Age", "38")
            fill_number(page, "Years of education", "13")
            fill_number(page, "Capital gain ($)", "0")
            fill_number(page, "Capital loss ($)", "0")
            fill_number(page, "Hours per week", "50")

            # Fill categorical selects
            select_option(page, "Workclass", "Private")
            select_option(page, "Marital status", "Married-civ-spouse")
            select_option(page, "Occupation", "Exec-managerial")
            select_option(page, "Relationship", "Husband")
            select_option(page, "Race", "White")
            select_option(page, "Sex", "Male")
            select_option(page, "Native country", "United-States")

            page.screenshot(path=str(SCREENSHOTS / "final_execution_4_form_filled.png"))
            log(4, "filled prediction form")

            page.get_by_role("button", name="Predict").click()

            # Wait for result header
            result_header = page.get_by_text("Result", exact=False).first
            result_header.wait_for(state="visible", timeout=15000)

            page.screenshot(path=str(SCREENSHOTS / "final_execution_5_prediction.png"))
            log(5, "prediction result rendered")

            prediction_text = result_header.locator("xpath=..").inner_text()
            log(6, f"rendered prediction text: {prediction_text[:200]}")

            browser.close()

        severe_errors = [
            e for e in console_errors if e["type"] in ("error", "pageerror")
        ]
        log(
            7,
            f"collected {len(console_errors)} console messages, {len(severe_errors)} severe",
        )

        with (RUN_DIR / "console_errors.json").open("w", encoding="utf-8") as f:
            json.dump(severe_errors, f, indent=2)

        if severe_errors:
            log(8, "WARNING: severe console errors detected")
        else:
            log(8, "no severe console errors detected")

        with LOG.open("a", encoding="utf-8") as f:
            f.write("\nFINAL_RESPONSE: prediction rendered successfully\n")
            f.write(f"SEVERE_CONSOLE_ERRORS: {len(severe_errors)}\n")

        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log(9, "streamlit subprocess terminated")


if __name__ == "__main__":
    sys.exit(main())
