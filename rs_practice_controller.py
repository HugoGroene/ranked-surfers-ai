#!/usr/bin/env python3
"""Record or test controls in a LOCAL Unity WebGL practice game.

Modes:
  inspect:   print actual HTML controls and canvas bounding boxes, no clicks
  probe:     verify HTML Practice button, canvas, and keyboard events
  calibrate: save stable-text-region screenshots of the animated start screen
  detect:    report start-screen confidence without clicking or restarting
  collect:   record manual gameplay and automatically restart practice
  random:    test synthetic arrow-key controls and automatic restarts

Only opens --url (defaults to localhost). No multiplayer automation.
"""
import argparse
import asyncio
import csv
from collections import deque
from datetime import datetime
import io
import json
from pathlib import Path
import random
import re
import shutil
import time

import cv2
import numpy as np
try:
    import mss
except ImportError:
    mss = None
from PIL import Image, ImageOps
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ai_data"
TEMPLATE = DATA / "start_template.png"
TEMPLATE_INFO = DATA / "start_template.json"
TEMPLATES = DATA / "start_templates"
KEYS = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]


def parse_roi(value):
    try:
        roi = [float(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use x,y,width,height between 0 and 1") from exc
    if len(roi) != 4 or any(v < 0 or v > 1 for v in roi) or roi[2] <= 0 or roi[3] <= 0 or roi[0] + roi[2] > 1 or roi[1] + roi[3] > 1:
        raise argparse.ArgumentTypeError("ROI must be normalized x,y,width,height within the image")
    return roi


def crop_image(image, roi):
    w, h = image.size
    x, y, rw, rh = roi
    box = (round(x*w), round(y*h), round((x+rw)*w), round((y+rh)*h))
    return image.crop(box).resize((256, 160), Image.Resampling.BILINEAR)


def edge_array(image):
    rgb = np.asarray(image.convert("L"))
    return cv2.Canny(rgb, 65, 135)


def similarity(screenshot, reference_edges, roi):
    """Compare a TIGHT text ROI, never the entire animated Unity menu."""
    im = Image.open(io.BytesIO(screenshot)).convert("RGB")
    edges = edge_array(crop_image(im, roi))
    refs = reference_edges if isinstance(reference_edges, list) else [reference_edges]
    scores = []
    for ref in refs:
        if np.std(edges) < 2 or np.std(ref) < 2:
            continue
        score = float(cv2.matchTemplate(edges, ref, cv2.TM_CCOEFF_NORMED)[0, 0])
        if np.isfinite(score):
            scores.append(score)
    return max(scores, default=0.0)


def total_directory_bytes(folder):
    if not folder.exists():
        return 0
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


async def find_canvas(page, timeout=35):
    """Unity may place the canvas on the page or within arena.html iframe."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for frame in page.frames:
            try:
                loc = frame.locator("canvas").first
                if await loc.count() and await loc.is_visible():
                    return loc
            except Exception:
                pass
        await asyncio.sleep(0.4)
    raise RuntimeError("No visible <canvas> found. Check the local game and browser Console.")


async def describe_page(page, folder):
    """Use the actual DOM to report HTML controls and canvases, including iframes.

    Does not click anything or guess pixel coordinates. Bounding boxes are
    reported in page/viewport coordinates by Playwright.
    """
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"Top-level URL: {page.url}", f"Frames found: {len(page.frames)}"]
    for i, frame in enumerate(page.frames):
        lines.append(f"\nFRAME {i}: {frame.url}")
        try:
            matches = frame.locator(
                "button, a, [role='button'], input[type='button'], "
                "input[type='submit'], canvas, iframe"
            )
            limit = min(await matches.count(), 120)
            for j in range(limit):
                el = matches.nth(j)
                if not await el.is_visible():
                    continue
                desc = await el.evaluate("""el => ({
                    tag: el.tagName.toLowerCase(),
                    id: el.id || '',
                    cls: typeof el.className === 'string' ? el.className : '',
                    text: (el.innerText || el.value || '').trim().slice(0, 110),
                    aria: el.getAttribute('aria-label') || '',
                    title: el.getAttribute('title') || ''
                })""")
                box = await el.bounding_box()
                coords = (f"x={box['x']:.0f} y={box['y']:.0f} "
                          f"w={box['width']:.0f} h={box['height']:.0f}"
                          if box else "no box")
                lines.append(f"  {desc['tag']}: {desc['text']!r} "
                             f"id={desc['id']!r} aria={desc['aria']!r} "
                             f"class={desc['cls'][:65]!r} {coords}")
        except Exception as exc:
            lines.append(f"  Error scanning frame: {str(exc)[:160]}")
    report = folder / "dom_inspection.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:85]))
    print("Full HTML inspection saved:", report)
    try:
        screenshot = folder / "page_inspection.png"
        await page.screenshot(path=str(screenshot), timeout=12000)
        print("Page screenshot saved:", screenshot)
    except Exception as exc:
        print("Could not screenshot:", str(exc)[:140])


async def practice_targets(frame):
    """Find *HTML* Practice controls, including ones in an iframe."""
    candidates = [
        frame.get_by_role("button", name=re.compile(r"practice", re.I)),
        frame.get_by_role("link", name=re.compile(r"practice", re.I)),
        frame.locator("button:has-text('Practice')"),
        frame.locator("[role='button']:has-text('Practice')"),
        frame.locator("input[type='button'][value*='practice' i]"),
        frame.locator("input[type='submit'][value*='practice' i]"),
        frame.locator("[aria-label*='practice' i]"),
    ]
    for group in candidates:
        try:
            count = min(await group.count(), 15)
            for index in range(count):
                target = group.nth(index)
                if await target.is_visible() and await target.is_enabled():
                    yield target
        except Exception:
            # A frame might navigate during loading; try again on the next poll.
            continue


async def debug_practice(page, folder):
    """Save evidence instead of accidentally clicking Unity's multiplayer button."""
    folder.mkdir(parents=True, exist_ok=True)
    shot = folder / "practice_not_found.png"
    try:
        await page.screenshot(path=str(shot), full_page=False, timeout=15000)
        print("Saved diagnostic screenshot:", shot)
    except Exception as exc:
        print("Could not save diagnostic screenshot:", str(exc)[:160])

    lines = [f"Top-level URL: {page.url}", f"Frames: {len(page.frames)}"]
    for i, frame in enumerate(page.frames):
        lines.append(f"\nFrame {i} URL: {frame.url}")
        try:
            elements = frame.locator("button, a, [role='button'], input[type='button'], input[type='submit']")
            for j in range(min(await elements.count(), 60)):
                el = elements.nth(j)
                if not await el.is_visible():
                    continue
                label = (await el.inner_text(timeout=1500) if await el.evaluate("e => 'innerText' in e")
                         else await el.get_attribute("value")) or ""
                aria = await el.get_attribute("aria-label") or ""
                lines.append(f"  <{await el.evaluate('e => e.tagName.toLowerCase()')}> text={label[:100]!r} aria={aria[:80]!r}")
        except Exception as exc:
            lines.append(f"  (unable to inspect controls: {str(exc)[:120]})")
    report = folder / "visible_controls.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("Saved visible HTML controls:", report)
    print("\n".join(lines[:30]))


async def click_practice(page, timeout=90, debug_dir=None):
    """Wait for the REAL HTML Practice control; never click the Unity start screen."""
    print(f"Waiting up to {timeout}s for the HTML Practice button ...")
    print("If Chromium appears behind the terminal, use Alt+Tab to bring it forward.")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in page.frames:
            async for target in practice_targets(frame):
                try:
                    label = (await target.inner_text(timeout=1500)).strip()[:100] or "(unnamed)"
                    box = await target.bounding_box()
                    print(f"Found HTML Practice control: {label!r} in {frame.url}")
                    if box:
                        print(f"Practice bounding box: x={box['x']:.0f}, y={box['y']:.0f}, "
                              f"width={box['width']:.0f}, height={box['height']:.0f}")
                    await target.click(timeout=5000)
                    return
                except Exception:
                    # Page re-rendered or the element was temporarily covered.
                    pass
        await asyncio.sleep(0.6)
    if debug_dir:
        await debug_practice(page, debug_dir)
    raise RuntimeError(
        f"No visible HTML Practice button after {timeout}s. "
        "Check practice_not_found.png and visible_controls.txt in ai_data/. "
        "This may mean the local website has not finished loading or an API is missing."
    )


async def focus_canvas(canvas):
    try:
        # Do not scroll the page each time we give keyboard focus.
        await canvas.evaluate("el => {el.tabIndex = 0; el.focus({preventScroll: true});}")
    except Exception as exc:
        print("Could not focus canvas:", str(exc)[:120])



async def screenshot_canvas(page, canvas, image_type="jpeg", quality=76):
    """Capture canvas viewport pixels without element.screenshot auto-scrolling.

    Playwright's element.screenshot() scrolls into view and can make the game
    page appear to jump/fullscreen. The bounding box of an iframe element is
    expressed in top-level viewport coordinates, so page.screenshot(clip=...) can
    sample the visible canvas without altering scroll or focus.
    """
    box = await canvas.bounding_box()
    if not box or box["width"] < 40 or box["height"] < 40:
        raise RuntimeError("Canvas has no usable on-screen bounding box")
    viewport = page.viewport_size
    if viewport is None:
        # A no_viewport context uses the real Chromium window size, which
        # changes when we enter F11-like fullscreen via CDP.
        viewport = await page.evaluate(
            "({width: window.innerWidth, height: window.innerHeight})"
        )
    x0 = max(0, box["x"])
    y0 = max(0, box["y"])
    x1 = min(viewport["width"], box["x"] + box["width"])
    y1 = min(viewport["height"], box["y"] + box["height"])
    if x1 - x0 < box["width"] * 0.90 or y1 - y0 < box["height"] * 0.90:
        raise RuntimeError(
            "Game canvas is partially outside the viewport. Increase --width/--height "
            "or scroll the page manually once until the canvas is fully visible."
        )
    kwargs = {"type": image_type, "clip": {"x": x0, "y": y0,
                                           "width": x1 - x0, "height": y1 - y0},
              "timeout": 15000}
    if image_type == "jpeg":
        kwargs["quality"] = quality
    return await page.screenshot(**kwargs)


async def mss_region_for_canvas(page, canvas, grabber, args):
    """Map the iframe canvas bounding box to X11 screen pixels once.

    mss samples the *visible desktop*, not the Chromium compositor. Chrome
    must remain uncovered, and the user should verify capture_preview.jpg.
    """
    box = await canvas.bounding_box()
    if not box or box["width"] < 40 or box["height"] < 40:
        raise RuntimeError("Unity canvas has no usable bounding box")

    metrics = await page.evaluate("""() => ({
        x: window.screenX, y: window.screenY,
        outerW: window.outerWidth, outerH: window.outerHeight,
        innerW: window.innerWidth, innerH: window.innerHeight,
        scale: window.devicePixelRatio
    })""")
    scale = metrics["scale"]
    # On Chromium/Linux, window.screenX/Y refer to the top-left of the Chrome
    # window; innerHeight excludes tabs / URL bar. Account for those controls.
    side = max(0, (metrics["outerW"] - metrics["innerW"]) / 2)
    top = max(0, metrics["outerH"] - metrics["innerH"] - side)
    x = round((metrics["x"] + side + box["x"] + args.screen_x_offset) * scale)
    y = round((metrics["y"] + top + box["y"] + args.screen_y_offset) * scale)
    width = round(box["width"] * scale)
    height = round(box["height"] * scale)
    screen = grabber.monitors[0]
    if (x < screen["left"] or y < screen["top"] or
        x + width > screen["left"] + screen["width"] or
        y + height > screen["top"] + screen["height"]):
        raise RuntimeError(
            f"Unity screen capture rectangle ({x},{y},{width},{height}) goes "
            f"outside the desktop ({screen}). The browser should be maximized."
        )
    print(f"MSS crop: x={x}, y={y}, width={width}, height={height}, scale={scale}")
    print("Keep Chromium visible and unobstructed while using MSS.")
    return {"left": x, "top": y, "width": width, "height": height}


def screenshot_mss(grabber, region, image_type="jpeg", quality=76):
    """Capture desktop pixels without asking Chromium to take a screenshot."""
    shot = np.asarray(grabber.grab(region))
    bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    ext = ".jpg" if image_type == "jpeg" else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if image_type == "jpeg" else []
    success, packed = cv2.imencode(ext, bgr, params)
    if not success:
        raise RuntimeError("OpenCV could not encode the MSS frame")
    return packed.tobytes()


async def run(args):
    DATA.mkdir(exist_ok=True)
    if args.mode in ("collect", "random", "detect"):
        if not TEMPLATE.is_file() or not TEMPLATE_INFO.is_file():
            raise RuntimeError("Calibration missing. Run --mode calibrate first.")
        roi = json.loads(TEMPLATE_INFO.read_text())["roi"]
        # Multiple menu animation frames; the *text* is stable but the backdrop moves.
        ref_paths = sorted(TEMPLATES.glob("*.png")) if TEMPLATES.is_dir() else []
        if not ref_paths:
            ref_paths = [TEMPLATE]
        ref_edges = [edge_array(Image.open(path)) for path in ref_paths]
    else:
        ref_edges, roi = None, args.roi

    if args.capture_backend == "mss" and args.mode in ("detect", "random", "collect") and mss is None:
        raise RuntimeError("MSS is not installed. Run: python -m pip install mss")

    key_events = []
    t0 = time.monotonic()

    def key_callback(source, data):
        if isinstance(data, dict) and data.get("code") in KEYS:
            key_events.append((time.monotonic() - t0, data.get("event"), data.get("code")))

    async with async_playwright() as p:
        launch_args = ["--start-maximized"]
        try:
            browser = await p.chromium.launch(headless=False, args=launch_args)
        except Exception:
            system_chrome = shutil.which("chromium") or shutil.which("chromium-browser")
            if not system_chrome:
                raise
            print("Using system Chromium:", system_chrome)
            browser = await p.chromium.launch(headless=False, executable_path=system_chrome, args=launch_args)
        # Let the actual Chrome window determine the browser viewport.
        # A fixed Playwright viewport can leave the game letterboxed
        # even when the outer Chromium window is maximized.
        context = await browser.new_context(no_viewport=True)
        await context.expose_binding("__rs_key_event", key_callback)
        await context.add_init_script("""
          (() => {
            for (const event of ['keydown', 'keyup']) {
              window.addEventListener(event, e => {
                if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.code)) {
                  // Prevent browser/page scrolling when Unity's canvas has focus.
                  // Do not stopPropagation: Unity must still receive the key.
                  if (event === 'keydown' && document.activeElement?.tagName === 'CANVAS') {
                    e.preventDefault();
                  }
                  window.__rs_key_event({event, code:e.code}).catch(() => {});
                }
              }, true);
            }
          })();
        """)
        page = await context.new_page()
        # Launch flags alone may not maximize Playwright-created windows.
        # Maximize normally, keeping the tab bar and address bar visible.
        try:
            cdp = await context.new_cdp_session(page)
            window = await cdp.send("Browser.getWindowForTarget")
            await cdp.send("Browser.setWindowBounds", {
                "windowId": window["windowId"],
                "bounds": {"windowState": "maximized"},
            })
            await page.wait_for_timeout(400)
            bounds = await cdp.send("Browser.getWindowBounds", {
                "windowId": window["windowId"],
            })
            state = bounds["bounds"].get("windowState", "unknown")
            print(f"Chromium window state: {state} (requested maximized browser window)")
        except Exception as exc:
            print("WARNING: Could not maximize Chrome via CDP:", str(exc)[:250])
            print("Try Alt+F10 (Ubuntu) to maximize the browser window.")
        page.on("console", lambda m: print("[Browser error]", m.text[:350]) if m.type == "error" else None)
        page.on("pageerror", lambda e: print("[Page exception]", str(e)[:350]))
        await page.goto(args.url, wait_until="domcontentloaded", timeout=45000)
        await page.bring_to_front()
        print("Chromium opened:", page.url)
        print("If the window is behind your terminal, switch to it with Alt+Tab.")

        # MSS is initialized only for continuous capture. Calibration and probe
        # retain one-off Playwright screenshots so keyboard/menu input is simple.
        grabber = None
        if args.capture_backend == "mss" and args.mode in ("detect", "random", "collect"):
            try:
                grabber = mss.mss()
            except Exception as exc:
                raise RuntimeError(
                    "Could not capture X11 desktop via MSS; check $DISPLAY and "
                    "whether you are using Wayland. " + str(exc)
                ) from exc

        try:
            if args.mode == "inspect":
                folder = DATA / ("inspect_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
                print("Inspecting the real HTML DOM; not clicking anything.")
                print(f"Waiting {args.inspect_delay} seconds for the UI to render ...")
                await asyncio.sleep(args.inspect_delay)
                await describe_page(page, folder)
                await asyncio.to_thread(input, "Inspect the browser if needed; Enter to close ... ")
                return

            if args.mode == "probe":
                print("\nPROBE: test ONE movement key after gameplay is visibly ready.")
                print("Only localhost is allowed; never clicks Unity's 1v1 start screen.")
                print("This mode clicks Practice once. It NEVER restarts a run automatically.")
                probe_dir = DATA / ("probe_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
                probe_dir.mkdir(exist_ok=True)
                if args.manual_start:
                    print("MANUAL START: Click the HTML Practice button yourself in Chromium.")
                else:
                    try:
                        await click_practice(page, timeout=args.practice_timeout, debug_dir=probe_dir)
                    except RuntimeError:
                        print("\nThe browser will stay open for inspection.")
                        await asyncio.to_thread(input, "Press Enter to close Chromium ... ")
                        raise
                    print("[OK] Clicked visible HTML Practice button ONCE")

                print("Wait until the character is actively running in Practice.")
                await asyncio.to_thread(input, "Then press Enter HERE to send ONE arrow key ... ")
                canvas = await find_canvas(page)
                box = await canvas.bounding_box()
                if box:
                    print(f"Unity canvas bounding box: x={box['x']:.0f}, y={box['y']:.0f}, "
                          f"width={box['width']:.0f}, height={box['height']:.0f}")
                print("Canvas HTML:", (await canvas.evaluate("el => el.outerHTML"))[:400])
                await focus_canvas(canvas)
                before = probe_dir / "before.png"
                before.write_bytes(await screenshot_canvas(page, canvas, image_type="png"))
                print("[OK] Captured before image:", before)
                await page.keyboard.press(args.probe_key)
                print("Sent ONE key:", args.probe_key)
                await asyncio.sleep(0.4)
                after = probe_dir / "after.png"
                after.write_bytes(await screenshot_canvas(page, canvas, image_type="png"))
                print("[OK] Captured after image:", after)
                await asyncio.sleep(0.3)
                counts = {key: sum(ev == "keydown" and code == key
                                   for _, ev, code in key_events) for key in KEYS}
                print("Keydown events captured in game iframe:", counts)
                print("Check whether the character ACTUALLY responded to that one action.")
                print("Probe complete; no automatic restarting or fullscreen commands were sent.")
                return

            if args.mode == "calibrate":
                print("\nCALIBRATION: use the ANIMATED death/menu screen containing 'FIND RANKED MATCH'.")
                print("Click Practice yourself, let the character die, then keep the menu visible.")
                print("Press Enter in the TERMINAL only AFTER the menu is visible.")
                await asyncio.to_thread(input, "Press Enter to capture FIVE animation frames ... ")
                canvas = await find_canvas(page)
                TEMPLATES.mkdir(exist_ok=True)
                for old_file in TEMPLATES.glob("*.png"):
                    old_file.unlink()
                for i in range(5):
                    raw = await screenshot_canvas(page, canvas, image_type="png")
                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                    if i == 0:
                        image.save(DATA / "start_screen_full.png")
                        # Show which PART of the screen is being matched.
                        from PIL import ImageDraw
                        marked = image.copy()
                        w, h = marked.size
                        x, y, rw, rh = roi
                        ImageDraw.Draw(marked).rectangle(
                            (round(x*w), round(y*h), round((x+rw)*w), round((y+rh)*h)),
                            outline="red", width=5)
                        marked.save(DATA / "start_roi_preview.png")
                    cropped = crop_image(image, roi)
                    cropped.save(TEMPLATES / f"ref_{i:02d}.png")
                    if i == 0:
                        cropped.save(TEMPLATE)
                    await asyncio.sleep(0.35)
                TEMPLATE_INFO.write_text(json.dumps({"roi": roi, "comment":
                    "ROI should TIGHTLY frame the static FIND RANKED MATCH text; no animated background."}, indent=2))
                print("Saved", len(list(TEMPLATES.glob('*.png'))), "reference frames.")
                print("Check:", DATA / "start_roi_preview.png")
                print("The red rectangle must tightly cover the 'FIND RANKED MATCH' text.")
                print("If not, recalibrate with --roi x,y,width,height (normalized 0..1).")
                print("Then run --mode detect to test scores WITHOUT auto restarting.")
                return

            if args.mode == "detect":
                backend = "MSS desktop capture" if grabber else "Playwright browser screenshots"
                print(f"\nDETECTION TEST: capture backend = {backend}.")
                print("This mode still prints start-screen similarity; the DETECTOR itself is unchanged.")
                print("Click Practice in Chromium and allow the game to end naturally.")
                print("Compare scores WHILE RUNNING versus at the 'FIND RANKED MATCH' menu.")
                print("To stop: Ctrl+C. (The browser may close.)")
                canvas = await find_canvas(page)
                region = await mss_region_for_canvas(page, canvas, grabber, args) if grabber else None
                if args.wait_for_game:
                    print("\nWait for Unity to load, click Practice, and start running.")
                    await asyncio.to_thread(input,
                        "When RUNNING, press Enter here; then switch BACK to Chromium within 4 seconds ... ")
                    await asyncio.sleep(4)
                until = time.monotonic() + args.max_minutes * 60
                first_frame = True
                while time.monotonic() < until:
                    raw = screenshot_mss(grabber, region) if grabber else await screenshot_canvas(page, canvas)
                    if first_frame:
                        (DATA / "capture_preview.jpg").write_bytes(raw)
                        print("Captured preview:", DATA / "capture_preview.jpg")
                        print("IMPORTANT: open this image only AFTER the test ends; the viewer will cover Chromium during MSS capture.")
                        first_frame = False
                    score = similarity(raw, ref_edges, roi)
                    print(f"Start-screen similarity: {score:.3f}", flush=True)
                    await asyncio.sleep(args.detect_interval)
                return

            existing_size = total_directory_bytes(DATA)
            max_bytes = int(args.max_gb * (1024**3))
            if existing_size >= max_bytes:
                print(f"Stopped: ai_data/ already occupies {existing_size / 1024**3:.2f} GiB, at or above --max-gb {args.max_gb}.")
                print("Move/archive older sessions or raise --max-gb deliberately.")
                return

            session = DATA / ("session_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.mode)
            session.mkdir()
            frames_csv = (session / "frames.csv").open("w", newline="", buffering=1)
            events_csv = (session / "events.csv").open("w", newline="", buffering=1)
            frames_writer = csv.writer(frames_csv)
            events_writer = csv.writer(events_csv)
            frames_writer.writerow(["elapsed_seconds", "run", "frame", "image", "keydown_since_last_frame", "start_screen_similarity"])
            events_writer.writerow(["elapsed_seconds", "run", "event", "key"])
            print("\nDATA DIRECTORY:", session)
            print("Frame size: 640x360 JPEG; limit:", args.max_gb, "GiB TOTAL ai_data; time:", args.max_minutes, "minutes")
            print("Press Ctrl+C to stop. The program only clicks HTML Practice and sends arrow keys.")

            run_no, frames_written = 0, 0
            interval = 1.0 / args.fps
            last_restart = 0.0
            last_action = 0.0
            last_info = 0.0
            # Keep a short history to tolerate animated-menu match-score dips.
            menu_votes = deque(maxlen=5)
            menu_suspected = False
            menu_clear_frames = 0
            gameplay_confirmations = 0
            armed = False
            bytes_used = existing_size
            next_tick = time.monotonic()
            start_time = time.monotonic()
            canvas = None
            current_run = None

            async def start_practice():
                nonlocal run_no, last_restart, menu_suspected, menu_clear_frames
                nonlocal gameplay_confirmations, armed, canvas, current_run, last_action
                # Never carry a held arrow key into the menu or the next run.
                for key in KEYS:
                    try:
                        await page.keyboard.up(key)
                    except Exception:
                        pass
                await click_practice(page, timeout=args.practice_timeout, debug_dir=session)
                run_no += 1
                current_run = session / f"run_{run_no:04d}"
                current_run.mkdir(exist_ok=True)
                last_restart = time.monotonic()
                last_action = last_restart
                menu_votes.clear()
                menu_suspected = False
                menu_clear_frames = 0
                gameplay_confirmations = 0
                armed = False
                key_events.clear()  # Avoid applying menu key events to the next run.
                await asyncio.sleep(1.2)
                canvas = await find_canvas(page)
                await focus_canvas(canvas)
                print(f"[RUN {run_no}] Practice clicked. Canvas ready.")

            try:
                await start_practice()
                region = await mss_region_for_canvas(page, canvas, grabber, args) if grabber else None
                first_frame = True
                while (time.monotonic() - start_time) < args.max_minutes * 60:
                    if bytes_used >= max_bytes:
                        print("Storage cap reached; stopping recording.")
                        break
                    # Always leave at least 2 GiB free; prevents accidentally filling the drive.
                    if frames_written % max(10, args.fps * 5) == 0:
                        if shutil.disk_usage(DATA).free < 2 * 1024**3:
                            print("Less than 2 GiB free; stopping.")
                            break

                    try:
                        raw = screenshot_mss(grabber, region, image_type="jpeg", quality=72) if grabber else await screenshot_canvas(page, canvas, image_type="jpeg", quality=72)
                        if first_frame:
                            (DATA / "capture_preview.jpg").write_bytes(raw)
                            print("Preview of captured Unity canvas:", DATA / "capture_preview.jpg")
                            first_frame = False
                    except Exception as exc:
                        print("Canvas temporarily unavailable:", str(exc)[:140])
                        await asyncio.sleep(0.8)
                        canvas = await find_canvas(page)
                        if grabber:
                            region = await mss_region_for_canvas(page, canvas, grabber, args)
                        continue

                    now = time.monotonic()
                    elapsed = now - t0  # same origin as events.csv timestamps
                    score = similarity(raw, ref_edges, roi)
                    # Only act after observing genuine gameplay. This prevents
                    # keypresses on the Unity loading/menu screen.
                    playing_cutoff = max(0.05, min(0.12, args.threshold * 0.40))
                    if score < playing_cutoff:
                        gameplay_confirmations += 1
                    else:
                        gameplay_confirmations = 0
                    if not armed and gameplay_confirmations >= 5:
                        armed = True
                        print(f"[RUN {run_no}] Gameplay confirmed; random input enabled.")

                    # Pause controls at the FIRST plausible menu frame. Confirm
                    # death using >=3 of the last 5 frames above --threshold;
                    # consecutive matches fail when the menu background animates.
                    if armed:
                        menu_votes.append(score >= args.threshold)
                        suspect_cutoff = args.threshold * 0.65
                        if not menu_suspected and score >= suspect_cutoff:
                            menu_suspected = True
                            menu_clear_frames = 0
                            print(f"[RUN {run_no}] Possible death (score {score:.2f}); pausing inputs.")

                        if menu_suspected:
                            if score < playing_cutoff:
                                menu_clear_frames += 1
                            else:
                                menu_clear_frames = 0
                            # A brief visual false alarm: resume after
                            # several unmistakable gameplay frames.
                            if menu_clear_frames >= 3:
                                menu_suspected = False
                                menu_votes.clear()
                                print(f"[RUN {run_no}] Gameplay visible again; resuming inputs.")

                    if (armed and menu_suspected
                            and now - last_restart >= args.min_run_seconds
                            and sum(menu_votes) >= 3):
                        print(f"[RUN {run_no}] Death confirmed ({sum(menu_votes)}/5 recent matches, score {score:.2f}); restarting Practice.")
                        await start_practice()
                        if grabber:
                            region = await mss_region_for_canvas(page, canvas, grabber, args)
                        next_tick = time.monotonic() + interval
                        continue

                    if (args.mode == "random" and armed and not menu_suspected
                            and now - last_restart > 1.2
                            and now - last_action > args.random_interval):
                        choice = random.choice(KEYS + [None, None, None])
                        if choice:
                            try:
                                await page.keyboard.press(choice)
                            except Exception as exc:
                                print("Could not send input:", str(exc)[:120])
                        last_action = time.monotonic()

                    # Menu images are not valid action-training examples. Do
                    # not write them to frames.csv as if they were gameplay.
                    if menu_suspected:
                        key_events.clear()
                        next_tick += interval
                        if time.monotonic() - next_tick > 2:
                            next_tick = time.monotonic()
                        await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
                        continue

                    since_last_frame = key_events[:]
                    key_events.clear()
                    downs = [key for _, ev, key in since_last_frame if ev == "keydown"]
                    for tm, ev, key in since_last_frame:
                        events_writer.writerow([f"{tm:.4f}", run_no, ev, key])

                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    im = ImageOps.pad(im, (640, 360), method=Image.Resampling.BILINEAR, color="black")
                    out_file = current_run / f"frame_{frames_written:07d}.jpg"
                    im.save(out_file, "JPEG", quality=68, optimize=True)
                    bytes_used += out_file.stat().st_size
                    frames_writer.writerow([
                        f"{elapsed:.4f}", run_no, frames_written,
                        out_file.relative_to(session).as_posix(),
                        "+".join(sorted(set(downs))) if downs else "NONE",
                        f"{score:.3f}"
                    ])
                    frames_written += 1
                    if now - last_info > 5:
                        print(f"[RUN {run_no}] {frames_written} frames | {bytes_used / 1024**3:.2f} GiB total | start similarity {score:.2f}")
                        last_info = now
                    next_tick += interval
                    if time.monotonic() - next_tick > 2:
                        next_tick = time.monotonic()  # don't accumulate delay
                    await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
            finally:
                frames_csv.close()
                events_csv.close()
                print("\nSaved", frames_written, "frames in", session)
        finally:
            if grabber is not None:
                grabber.close()
            await browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["inspect", "probe", "calibrate", "detect", "collect", "random"], required=True)
    parser.add_argument("--url", default="http://localhost:8000", help="Local practice site URL")
    parser.add_argument("--roi", type=parse_roi, default=[0.30, 0.30, 0.40, 0.30], help="TIGHT 'FIND RANKED MATCH' text crop x,y,width,height 0..1; view ai_data/start_roi_preview.png")
    parser.add_argument("--width", type=int, default=1600, help="Initial width hint (maximized window uses screen dimensions)")
    parser.add_argument("--height", type=int, default=1000, help="Initial height hint (maximized window uses screen dimensions)")
    parser.add_argument("--capture-backend", choices=["mss", "playwright"], default="mss",
                        help="Capture for detect/collect/random; MSS avoids Chrome screenshot side effects (default mss)")
    parser.add_argument("--detect-interval", type=float, default=0.5,
                        help="Seconds between detector screenshots, default 0.5")
    parser.add_argument("--wait-for-game", action="store_true",
                        help="Detect mode: wait until you confirm Practice is running, then give you 4s to focus Chromium before capturing")
    parser.add_argument("--screen-x-offset", type=int, default=0,
                        help="Desktop capture x adjustment (CSS pixels), if preview misaligned")
    parser.add_argument("--screen-y-offset", type=int, default=0,
                        help="Desktop capture y adjustment (CSS pixels), if preview misaligned")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default 10)")
    parser.add_argument("--max-minutes", type=float, default=30.0, help="Max recording session duration")
    parser.add_argument("--max-gb", type=float, default=5.0, help="Max total size of ai_data in GiB")
    parser.add_argument("--threshold", type=float, default=0.82, help="Start-screen edge match threshold (0..1)")
    parser.add_argument("--min-run-seconds", type=float, default=5.0, help="Minimum seconds before death detection")
    parser.add_argument("--random-interval", type=float, default=0.7, help="Time between random control tests")
    parser.add_argument("--inspect-delay", type=float, default=8.0,
                        help="Wait this many seconds before reading HTML in --mode inspect")
    parser.add_argument("--practice-timeout", type=int, default=90,
                        help="Seconds to wait for HTML Practice button (default: 90)")
    parser.add_argument("--manual-start", action="store_true",
                        help="Probe mode only: let you click Practice manually before testing keys")
    parser.add_argument("--probe-key", choices=KEYS, default="ArrowLeft",
                        help="Probe mode: press only this one key after you confirm gameplay has begun")
    args = parser.parse_args()
    if not (800 <= args.width <= 3840 and 700 <= args.height <= 2160 and 1 <= args.fps <= 30 and 0.05 <= args.detect_interval <= 60 and 0 < args.max_minutes <= 1440 and 0 < args.max_gb <= 10000 and 0 < args.threshold <= 1 and args.min_run_seconds >= 0 and 1 <= args.practice_timeout <= 600 and 0 <= args.inspect_delay <= 600):
        parser.error("Invalid FPS, duration, storage cap, threshold or minimum run duration")
    if not args.url.startswith(("http://localhost:", "http://127.0.0.1:")):
        parser.error("For safety this starter controller is limited to your LOCAL practice copy (localhost).")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (RuntimeError, Exception) as exc:
        print("\nERROR:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
