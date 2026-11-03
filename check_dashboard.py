"""Browser smoke check against a disposable local server, never live user data.

Install Playwright Chromium first, or set BROWSER_EXECUTABLE to a local Chromium/Edge binary.
"""
import json
import os
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server, WSGIRequestHandler
import test_runtime


class QuietHandler(WSGIRequestHandler):
    def log(self, *args, **kwargs):
        pass


def run():
    fixture = test_runtime.RuntimeTests()
    fixture.setUp()
    server = None
    try:
        for name in ("index.html", "workspace.js", "workspace.css"):
            shutil.copyfile(Path(__file__).with_name(name), fixture.root / name)
        app = fixture.app
        app.ai = SimpleNamespace(label="test")
        app.settings = app.clean_settings({"cameras": [{"id": "fixture", "name": "Test camera", "source": "0", "enabled": False}]})
        server = make_server("127.0.0.1", 0, app.app, threaded=True, request_handler=QuietHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        with sync_playwright() as playwright:
            options = {"headless": True}
            if os.environ.get("BROWSER_EXECUTABLE"):
                options["executable_path"] = os.environ["BROWSER_EXECUTABLE"]
            browser = playwright.chromium.launch(**options)
            context = browser.new_context(http_credentials={"username": "admin", "password": "admin"},
                                          viewport={"width": 1280, "height": 900})
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.wait_for_function("S !== null && ST !== null")
            page.evaluate("showTab('settings')")
            page.locator("#schedule-enabled").check()
            page.locator("#schedule-start").fill("20:00")
            page.locator("#schedule-end").fill("06:00")
            page.locator("#schedule-timezone").fill("Asia/Singapore")
            page.locator('[data-settings="notifications"]').click()
            page.locator("#offline_after").fill("45")
            page.locator("#alert_min_seconds").fill("3")
            page.locator('[data-settings="recording"]').click()
            page.locator("#storage_cleanup").check()
            page.locator("#retention_days").fill("14")
            page.locator("#min_free_gb").fill("2")
            page.locator("#save").click()
            page.wait_for_function("document.querySelector('#save-msg').textContent.includes('Saved and applied')")
            assert app.settings["offline_after"] == 45
            assert app.settings["alert_min_seconds"] == 3
            assert app.settings["storage_cleanup"] is True
            assert app.settings["min_free_gb"] == 2
            assert app.settings["cameras"][0]["schedule"]["timezone"] == "Asia/Singapore"
            assert app.settings["cameras"][0]["schedule"]["start"] == "20:00"
            page.locator('[data-settings="login"]').click()
            with page.expect_download() as download_info:
                page.locator('a[href="/api/backup"]').click()
            backup = Path(download_info.value.path())
            assert json.loads(backup.read_text())["settings"]["retention_days"] == 14
            page.locator("#backup-file").set_input_files(backup)
            page.wait_for_function("!document.querySelector('#restore-backup').disabled")
            app.settings["retention_days"] = 99
            page.on("dialog", lambda dialog: dialog.accept())
            with page.expect_navigation(wait_until="domcontentloaded"):
                page.locator("#restore-backup").click()
            page.wait_for_function("S !== null && S.retention_days === 14")
            assert app.settings["retention_days"] == 14
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate("showTab('settings')")
            page.locator('[data-settings="notifications"]').click()
            assert page.locator("#offline_after").is_visible()
            assert not errors, errors
            context.close()
            browser.close()
        print("Dashboard browser check passed: schedules, alerts, retention, backup/restore and mobile controls")
    finally:
        if server:
            server.shutdown()
            server.server_close()
        fixture.doCleanups()


if __name__ == "__main__":
    run()
