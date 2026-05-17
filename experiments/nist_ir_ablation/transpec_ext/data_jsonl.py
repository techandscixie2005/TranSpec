import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class IrSpectrumRecord:
    idx: int
    smiles: str
    wavenumbers: List[float]
    intensities: List[float]
    condition: Optional[str] = None
    source: Optional[str] = None
    raw: Optional[dict] = None


def read_jsonl(path: str, min_points: int = 100) -> Tuple[List[IrSpectrumRecord], List[dict]]:
    """
    Read a JSONL file of IR spectra records.

    Each line must contain at minimum:
        "smiles": str
        "value": {"x": [float, ...], "y": [float, ...]}

    Returns (records, invalid_entries) where invalid_entries is a list of
    dicts with keys: line_number, reason, (and optionally "content").
    """
    records = []
    invalid = []

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # 1. Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                invalid.append({
                    "line_number": line_number,
                    "reason": f"Invalid JSON: {e}",
                })
                continue

            # 2. Check required fields
            if "smiles" not in data:
                invalid.append({
                    "line_number": line_number,
                    "reason": "Missing 'smiles' field",
                })
                continue

            if "value" not in data or not isinstance(data["value"], dict):
                invalid.append({
                    "line_number": line_number,
                    "reason": "Missing or invalid 'value' field",
                })
                continue

            if "x" not in data["value"] or "y" not in data["value"]:
                invalid.append({
                    "line_number": line_number,
                    "reason": "Missing 'value.x' or 'value.y'",
                })
                continue

            x = data["value"]["x"]
            y = data["value"]["y"]

            if not isinstance(x, list) or not isinstance(y, list):
                invalid.append({
                    "line_number": line_number,
                    "reason": "'value.x' or 'value.y' is not a list",
                })
                continue

            if len(x) != len(y):
                invalid.append({
                    "line_number": line_number,
                    "reason": f"Mismatched x/y length: {len(x)} vs {len(y)}",
                })
                continue

            if len(x) < min_points:
                invalid.append({
                    "line_number": line_number,
                    "reason": f"Too few points: {len(x)} < {min_points}",
                })
                continue

            # 3. Extract optional metadata
            raw = {}
            for optional_key in ("temperature", "pressure", "condition", "type", "source"):
                if optional_key in data:
                    raw[optional_key] = data[optional_key]

            source = data.get("source")
            condition = data.get("condition")

            record = IrSpectrumRecord(
                idx=line_number - 1,
                smiles=data["smiles"],
                wavenumbers=[float(v) for v in x],
                intensities=[float(v) for v in y],
                condition=condition,
                source=source,
                raw=raw if raw else None,
            )
            records.append(record)

    return records, invalid
