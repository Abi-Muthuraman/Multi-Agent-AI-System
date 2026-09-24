"""
Requester Agent
----------------
Coordinates the workflow using the real A2A protocol (Aadi) and
the real Specialist Agent / RAG pipeline (Abi):
1. Takes the user's request
2. Submits it to the Specialist Agent via A2ATaskManager
3. Polls until completed / failed / timeout
4. Maps the result into form field values for Playwright
"""

from a2a_protocol import A2ATaskManager
from specialist_agent import SpecialistAgent

# Maps the Specialist's category to the 4 options in the mock form's dropdown.
# The form has no Email or Security option, so those fall back to the closest fit.
CATEGORY_MAP = {
    "Account Access": "account_access",
    "Hardware": "hardware",
    "Software": "software",
    "Network": "network",
    "Email": "account_access",     # no Email option in form, closest fit
    "Security": "account_access",  # no Security option in form, closest fit
    "Password Reset": "account_access", # maps directly to Account Access
}


class RequesterAgent:
    def __init__(self, timeout=15.0, poll_interval=0.5):
        self.manager = A2ATaskManager()
        self.specialist = SpecialistAgent()  # loads embeddings + connects to Groq
        self.timeout = timeout
        self.poll_interval = poll_interval

    def handle_request(self, user_request):
        """
        Runs the full workflow for one user request.
        Returns a dict of form-ready fields, or a fields dict signaling failure.
        """
        ack = self.manager.submit_task(user_request, self.specialist.answer)
        print(f"Task submitted: {ack['task_id']} | status: {ack['status']}")

        final = self.manager.poll_until_done(
            ack["task_id"], timeout=self.timeout, interval=self.poll_interval
        )
        print(f"Final task status: {final['status']}")

        return self._build_form_fields(final, user_request)

    def _build_form_fields(self, final, user_request):
        status = final["status"]

        if status == "completed":
            result = final["result"]
            raw_category = result.get("category", "Unknown")
            mapped_category = CATEGORY_MAP.get(raw_category, "software")  # default fallback
            return {
                "issue": user_request,
                "category": mapped_category,
                "resolution_notes": result.get("resolution", ""),
                "ok": True,
            }

        if status == "failed":
            print(f"Specialist Agent failed: {final.get('error')}")
        elif status == "timeout":
            print(f"Specialist Agent timed out: {final.get('error')}")

        return {
            "issue": user_request,
            "category": None,
            "resolution_notes": "No resolution retrieved. Escalate for manual review.",
            "ok": False,
        }


if __name__ == "__main__":
    agent = RequesterAgent(timeout=15.0)

    print("=== Success case ===")
    fields = agent.handle_request("I forgot my password and cannot log into my account.")
    print(fields)

    print("\n=== Timeout case (force it with a tiny timeout) ===")
    fast_agent = RequesterAgent(timeout=0.01)
    fields2 = fast_agent.handle_request("My laptop won't connect to Wi-Fi.")
    print(fields2)