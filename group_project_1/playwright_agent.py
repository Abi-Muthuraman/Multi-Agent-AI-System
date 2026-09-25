"""
Playwright Automation
----------------------
Takes the Requester Agent's output (issue, category, resolution_notes)
and fills out the mock support ticket form, submits it, and verifies
the confirmation appears. If the Requester Agent failed or timed out,
skips the form fill and logs it as a failure instead of crashing.
"""

import json
from pathlib import Path
from playwright.sync_api import sync_playwright

from requester_agent import RequesterAgent

FORM_PATH = Path(__file__).parent / "mock_support_app" / "index.html"
FORM_URL = f"file://{FORM_PATH.resolve()}"


def submit_ticket(page, fields):
    """Fills the form and submits. Returns True if confirmation appeared."""
    page.goto(FORM_URL)

    page.fill("#issue", fields["issue"])
    page.select_option("#category", fields["category"])
    page.fill("#resolution", fields["resolution_notes"])
    page.click("#submit-ticket")

    try:
        page.wait_for_selector("#confirmation:not(.hidden)", timeout=3000)
        ticket_id = page.text_content("#ticket-id")
        category_shown = page.text_content("#ticket-category")
        print(f"Ticket submitted. ID: {ticket_id}, Category: {category_shown}")
        return True
    except Exception:
        print("Confirmation did not appear, submission may have failed.")
        return False


def run_case(agent, browser, user_request):
    fields = agent.handle_request(user_request)

    if not fields["ok"]:
        print(f"Skipping form fill: {fields['resolution_notes']}")
        return {"request": user_request, "submitted": False, "reason": fields["resolution_notes"]}

    page = browser.new_page()
    submitted = submit_ticket(page, fields)
    page.wait_for_timeout(2000)  # pause 2 seconds so you can see the result
    page.close()

    return {"request": user_request, "submitted": submitted, "category": fields["category"]}


def run_all_test_cases():
    with open(Path(__file__).parent / "test_cases.json") as f:
        test_cases = json.load(f)

    agent = RequesterAgent(timeout=15.0)
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=900)  # set True once you trust it

        for case in test_cases:
            print(f"\n=== Test case {case['id']}: {case['request']} ===")
            result = run_case(agent, browser, case["request"])
            results.append(result)

        browser.close()

    print("\n=== Summary ===")
    for r in results:
        print(r)

    return results


def run_failure_demo():
    """Forces a timeout so you have a real failure case for the demo video."""
    print("\n=== Forced timeout demo ===")
    fast_agent = RequesterAgent(timeout=0.01)
    fields = fast_agent.handle_request("My laptop won't connect to Wi-Fi.")
    print(fields)


if __name__ == "__main__":
    run_all_test_cases()
    run_failure_demo()