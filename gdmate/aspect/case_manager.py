import pandas as pd
from pathlib import Path


class CaseManager:
    """
    Minimal manager for numerical cases using a CSV file.

    CSV format:
        case_id,case_path
    """

    def __init__(self, csv_file):
        self.csv_file = Path(csv_file)
        self.df = pd.read_csv(self.csv_file)

        # normalize paths
        self.df["case_path"] = self.df["case_path"].apply(
            lambda p: str(Path(p).resolve())
        )

        # ensure unique case IDs
        assert self.df["case_id"].is_unique, "case_id must be unique"

    def _get_row(self, mask, error_msg):
        """Internal helper returning (row_index, row)."""
        subset = self.df[mask]

        if subset.empty:
            raise ValueError(error_msg)

        idx = subset.index[0]
        row = subset.iloc[0]

        return idx, row

    def get_case_by_id(self, case_id):
        """Return (row_index, row) for a given case ID."""
        return self._get_row(
            self.df["case_id"] == case_id,
            f"Case {case_id} not found"
        )

    def get_case_by_path(self, case_path):
        """Return (row_index, row) for a given case path."""
        case_path = str(Path(case_path).resolve())

        return self._get_row(
            self.df["case_path"] == case_path,
            f"Case with path '{case_path}' not found"
        )

    def list_cases(self):
        """Return full dataframe."""
        return self.df