
import threading
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

PROD_SHEET = "Production"

PROD_COLUMNS = [
    "Timestamp",
    "User Query",
    "Bot Response",
    "Time Taken (s)",
    "Session ID",
    "Source",
]


class ResponseLogger:
    """Thread-safe Excel response logger."""

    def __init__(self, log_dir=None, filename="response_log.xlsx"):
        """
        Initialize logger with thread safety
        (single multi-sheet Excel file).
        """
        self.lock = threading.Lock()

        self.log_dir = (
            Path(log_dir)
            if log_dir
            else Path(__file__).resolve().parent.parent.parent / "logs"
        )

        self.log_file = self.log_dir / filename

        self._ensure_log_file()

    def _make_fresh_wb(self):
        """Create a brand new workbook with both sheets."""

        wb = Workbook()

        ws_prod = wb.active
        ws_prod.title = PROD_SHEET
        ws_prod.append(PROD_COLUMNS)

        for cell in ws_prod[1]:
            cell.font = Font(bold=True)

        ws_eval = wb.create_sheet("Evaluation")

        eval_cols = [
            "Timestamp",
            "Prompt",
            "Bot Answer",
            "Source",
            "Retrieval Confidence (%)",
            "Latency (s)",
            "Server Time (s)",
            "Faithfulness % (LLM)",
            "Relevance % (LLM)",
            "Completeness % (LLM)",
            "BERTScore F1 %",
            "Link Validity",
            "Accuracy %",
        ]

        ws_eval.append(eval_cols)

        for cell in ws_eval[1]:
            cell.font = Font(bold=True)

        return wb

    def _ensure_log_file(self):
        """Ensure log file exists and sheets are valid."""

        with self.lock:

            self.log_dir.mkdir(parents=True, exist_ok=True)

            if not self.log_file.exists():

                wb = self._make_fresh_wb()
                wb.save(self.log_file)

            else:

                try:
                    wb = load_workbook(self.log_file)

                    if PROD_SHEET not in wb.sheetnames:

                        ws = wb.create_sheet(PROD_SHEET, 0)
                        ws.append(PROD_COLUMNS)

                        for cell in ws[1]:
                            cell.font = Font(bold=True)

                        wb.save(self.log_file)

                except Exception as exc:

                    print(f"⚠ Log file corrupted ({exc}). Resetting...")

                    backup = self.log_file.with_suffix(
                        f".corrupt_{int(datetime.now().timestamp())}.xlsx"
                    )

                    try:
                        self.log_file.rename(backup)
                    except Exception:
                        pass

                    self._make_fresh_wb().save(self.log_file)

    def log_response(
        self,
        user_query,
        bot_response,
        time_taken,
        session_id="N/A",
        source="N/A",
    ):
        """Append a row to Production sheet (thread safe)."""

        with self.lock:

            try:

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                try:

                    if self.log_file.exists():
                        wb = load_workbook(self.log_file)
                    else:
                        wb = self._make_fresh_wb()

                except Exception as exc:

                    print(
                        f"⚠ Error loading log ({exc}). Resetting..."
                    )
                    wb = self._make_fresh_wb()

                if PROD_SHEET in wb.sheetnames:
                    ws = wb[PROD_SHEET]
                else:
                    ws = wb.create_sheet(PROD_SHEET, 0)
                    ws.append(PROD_COLUMNS)

                ws.append(
                    [
                        timestamp,
                        user_query,
                        bot_response,
                        f"{time_taken:.4f}",
                        session_id,
                        source,
                    ]
                )

                wb.save(self.log_file)

                print(
                    f"✓ Logged response to: "
                    f"{self.log_file} "
                    f"[Sheet: {PROD_SHEET}]"
                )

            except PermissionError:

                print(
                    f"⚠ ERROR: Could not write to {self.log_file}. "
                    "Is it open in Excel?"
                )

            except Exception as exc:

                print(f"Error logging response: {exc}")
