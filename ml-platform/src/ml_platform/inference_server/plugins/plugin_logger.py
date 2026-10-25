import csv
import logging

from ml_platform.inference_server.plugins.plugin import PluginResult
from ml_platform.shared.utils import init_log_path

logger = logging.getLogger(__name__)

class PluginLogger():
    def __init__(self):
        self.records = {}
        self.flush_rows = 20_000
        self.pending_updates = 0
        self.required_keys = ("sample_index", "time_stamp")

    def init_log(self):
        self.log_file = init_log_path("plugin_log", "csv")
        logger.info(f"Plugin log file: {self.log_file}")

    def collect_results(self, results: list[PluginResult]):

        for result in results:
            self.collect(result)


    def collect(self, result: PluginResult):
        metrics = result.metrics
        name = result.name

        missing = set(self.required_keys) - metrics.keys()
        if missing:
            logger.warning( f"Results from plugin '{name}' missing required keys: {missing}" )
            return

        idx = metrics["sample_index"]

        prefixed = {}

        for k, v in metrics.items():
            if k in self.required_keys:
                prefixed[k] = v
            else:
                prefixed[f"{name}.{k}"] = v

        metrics = prefixed

        if idx not in self.records:
            self.records[idx] = metrics.copy()
        else:
            self.records[idx].update(metrics)

        self.pending_updates += 1

        if self.pending_updates >= self.flush_rows:
            self.flush_and_init()
            self.pending_updates = 0


    def flush(self):
        if not self.records:
            return

        # Collect all field names that have ever appeared
        fieldnames = set()
        for record in self.records.values():
            fieldnames.update(record.keys())

        # Keep required fields first, then sort the rest
        fieldnames = list(self.required_keys) + sorted(
            fieldnames - set(self.required_keys)
        )

        # Write the entire table
        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            for record in sorted( self.records.values(), key=lambda r: r["sample_index"] ):
                writer.writerow(record)

        logger.info( f"Flushed {len(self.records)} records to {self.log_file}" )
        self.records = {}

    def flush_and_init(self):
        file_name = self.log_file
        self.flush()
        self.init_log()
        return file_name