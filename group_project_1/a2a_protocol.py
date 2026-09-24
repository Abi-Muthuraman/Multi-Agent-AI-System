import time
import uuid
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Dict

class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    task_id:str
    request: str
    status: TaskStatus = TaskStatus.SUBMITTED
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class A2ATaskManager:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()

    def submit_task(self, request: str, handler: Callable[[str], Any]) -> dict:
        task_id = str(uuid.uuid4())[:8]
        task = Task(task_id=task_id, request=request)
        with self._lock:
            self._tasks[task_id] = task
 
        thread = threading.Thread(target=self._run, args=(task_id, handler), daemon=True)
        thread.start()
 
        return {"task_id": task_id, "status": task.status.value}

    
    def _run(self, task_id: str, handler: Callable[[str], Any]) -> None:
        self._set_status(task_id, TaskStatus.WORKING)
        task = self._tasks[task_id]
        try:
            result = handler(task.request)
            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.updated_at = time.time()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any handler failure -> "failed"
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                task.updated_at = time.time()

    def _set_status(self, task_id: str, status: TaskStatus) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = status
                task.updated_at = time.time()

    def get_status(self, task_id: str) -> dict:
        """Returns just {"task_id", "status"} — cheap check, no result payload."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {"task_id": task_id, "status": "not_found"}
            return {"task_id": task_id, "status": task.status.value}


    def get_result(self, task_id: str) -> dict:
        """
        Returns the full task record. Always check "status" before trusting
        "result" — it's only populated once status == "completed".
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {"task_id": task_id, "status": "not_found", "result": None, "error": None}
            return {
                "task_id": task_id,
                "status": task.status.value,
                "result": task.result,
                "error": task.error,
            }

    def poll_until_done(self, task_id: str, timeout: float = 15.0, interval: float = 0.5) -> dict:
        """
        Polls get_result() until status is "completed"/"failed" or `timeout`
        seconds pass. On timeout, returns status "timeout" instead of hanging
        forever — this is the Requester Agent's timeout/failure handling.
        """
        start = time.time()
        while time.time() - start < timeout:
            result = self.get_result(task_id)
            if result["status"] in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
                return result
            time.sleep(interval)
 
        return {
            "task_id": task_id,
            "status": "timeout",
            "result": None,
            "error": f"Task did not complete within {timeout} seconds.",
        }
    

if __name__ == "__main__":

    def fake_specialist_handler(request_text: str) -> dict:
        """
        Stand-in for Abi's real RAG pipeline. Replace this with her function
        once it's ready — the manager doesn't care what's inside, only that
        it returns a dict or raises.
        """
        time.sleep(2)  # simulate retrieval + LLM latency
        return {
            "category": "Account Access",
            "resolution": "Verify the user's identity, then reset their password.",
        }

    def fake_failing_handler(request_text: str) -> dict:
        raise RuntimeError("No relevant documents found in knowledge base.")
 
    manager = A2ATaskManager()
 
    print("--- Success case ---")
    ack = manager.submit_task("I forgot my password.", fake_specialist_handler)
    print("Ack:", ack)
    print("Immediate status check:", manager.get_status(ack["task_id"]))  # likely "working"
    final = manager.poll_until_done(ack["task_id"], timeout=10)
    print("Final:", final)
 
    print("\n--- Failure case ---")
    ack2 = manager.submit_task("garbled nonsense request", fake_failing_handler)
    final2 = manager.poll_until_done(ack2["task_id"], timeout=10)
    print("Final:", final2)
 
    print("\n--- Timeout case (handler takes longer than the timeout) ---")


    def slow_handler(request_text: str) -> dict:
        time.sleep(5)
        return {"category": "Hardware", "resolution": "Replace the keyboard."}
 
    ack3 = manager.submit_task("my keyboard is dead", slow_handler)
    final3 = manager.poll_until_done(ack3["task_id"], timeout=2, interval=0.5)
    print("Final:", final3)
    
    

    

    
    
 
 