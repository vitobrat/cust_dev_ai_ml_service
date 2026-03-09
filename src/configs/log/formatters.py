from collections import OrderedDict
from typing import Any, Dict

from pythonjsonlogger import jsonlogger


class LogsJsonFormatter(jsonlogger.JsonFormatter):
    def process_log_record(self, log_record: Dict[str, Any]) -> OrderedDict[str, Any]:
        super().process_log_record(log_record)
        rec = OrderedDict(
            {
                "level": log_record.get("levelname"),
                "datetime": log_record.get("asctime"),
                "msg": log_record.get("message"),
                "module": log_record.get("module"),
                "line": log_record.get("lineno"),
            },
        )
        if log_record.get("exc_info") is not None:
            rec["exc_info"] = log_record.get("exc_info")
        return rec
