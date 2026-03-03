import json
import pathlib
from datetime import datetime, timezone
import numpy as np
import polars as pl

class ExplainabilityBundle:
    """
    Saves all artifacts produced during a run.
    """
    def __init__(self, output_dir: str, run_id: str):
        self.output_dir = pathlib.Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.sections = []

    def save_config(self, config_dict: dict):
        path = self.output_dir / "run_config.json"
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def save_array(self, name: str, array: np.ndarray):
        path = self.output_dir / f"{name}.npy"
        np.save(path, array)

    def save_table(self, name: str, df: pl.DataFrame):
        path = self.output_dir / f"{name}.parquet"
        df.write_parquet(path)

    def add_section(self, title: str, content: str):
        self.sections.append(f"## {title}\n\n{content}\n")

    def write_math_report(self):
        report_path = self.output_dir / "math_report.md"
        with open(report_path, "w") as f:
            f.write(f"# XVA Engine Math Report\n\n")
            f.write(f"Run ID: `{self.run_id}`\n")
            f.write(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write("---\n\n")
            for section in self.sections:
                f.write(section)
                f.write("\n---\n\n")
