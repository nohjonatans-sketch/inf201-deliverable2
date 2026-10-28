"""Deliverable 2: Campus Coffee Shop sales pipeline."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

LINE_PATTERN = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] "
    r"\[(?P<register_id>\w+)\] "
    r"item=(?P<item>\w+) "
    r"price=(?P<price>\d+(?:\.\d+)?) "
    r"(?P<currency>[A-Z]{3}) "
    r"\((?P<status>SUCCESS|REFUND|CANCELLED)\)$"
)


def read_log_file(path: Path | str) -> tuple[str, str]:
    """Read a text file, trying UTF-8 first and then Latin-1."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("iso-8859-1"), "iso-8859-1"


def parse_log(text: str) -> list[dict]:
    """Return one dict per valid sales line; noise lines are ignored."""
    records = []
    for line in text.splitlines():
        match = LINE_PATTERN.match(line.strip())
        if match:
            record = match.groupdict()
            record["price"] = float(record["price"])
            records.append(record)
    return records


def find_log_files(folder: Path | str) -> list[Path]:
    """Recursively find all .log files below folder."""
    folder = Path(folder)
    log_files = []
    for item in folder.iterdir():
        if item.is_dir():
            log_files.extend(find_log_files(item))
        elif item.is_file() and item.suffix == ".log":
            log_files.append(item)
    return log_files


def find_active_registers(data: dict | list) -> set[str]:
    """Recursively collect the ids of all registers with "active": true."""
    active = set()
    if isinstance(data, dict):
        if data.get("active") is True and "id" in data:
            active.add(data["id"])
        for value in data.values():
            active |= find_active_registers(value)
    elif isinstance(data, list):
        for item in data:
            active |= find_active_registers(item)
    return active


def remove_outliers(prices) -> np.ndarray:
    """Replace values more than 3 standard deviations from the mean with the median of the valid values."""
    prices = np.asarray(prices, dtype=float).ravel()
    is_outlier = np.abs(prices - prices.mean()) > 3 * prices.std()
    cleaned = prices.copy()
    cleaned[is_outlier] = np.median(prices[~is_outlier])
    return cleaned


def compute_refund_ratio(sales_count: np.ndarray, refund_count: np.ndarray) -> np.ndarray:
    """Refund ratio = refunds / (sales + 1e-6), element-wise."""
    return refund_count / (sales_count + 1e-6)


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Make a DataFrame with a sorted datetime index from the parsed records."""
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def hourly_revenue(df: pd.DataFrame) -> pd.Series:
    """Total price per item in 1-hour windows."""
    return df.groupby("item")["price"].resample("1h").sum()


def status_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Total revenue per item for each status."""
    return df.pivot_table(
        values="price", index="item", columns="status", aggfunc="sum", fill_value=0
    )


def item_refund_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Refund ratio per item (refund count / successful sales count)."""
    counts = df.pivot_table(
        values="price", index="item", columns="status", aggfunc="count", fill_value=0
    )
    ratios = compute_refund_ratio(
        counts["SUCCESS"].to_numpy(), counts["REFUND"].to_numpy()
    )
    return pd.DataFrame({"refund_ratio": ratios}, index=counts.index)


# 1. Find log files and active registers
log_files = find_log_files(DATA_DIR)
with open(DATA_DIR / "store_config.json", encoding="utf-8") as f:
    active_registers = find_active_registers(json.load(f))
print(f"Found {len(log_files)} log files, active registers: {sorted(active_registers)}")

# 2. Read and parse the logs (only active registers)
records = []
for log_file in log_files:
    text, encoding = read_log_file(log_file)
    records += [r for r in parse_log(text) if r["register_id"] in active_registers]
print(f"Parsed {len(records)} transactions")

# 3. Build DataFrame
df = build_dataframe(records)

# 4. Remove 999.00 outliers
df["price"] = remove_outliers(df["price"])

# 5. Currency correction: NOK prices were registered in EUR scale
df.loc[df["currency"] == "NOK", "price"] *= 10.0

# 6 + 7. Reports and output files
df.to_csv("cleaned_transactions.csv")

revenue = hourly_revenue(df)
revenue_dict = {
    item: {str(time): round(value, 2) for time, value in group.droplevel(0).items()}
    for item, group in revenue.groupby(level=0)
}
with open("hourly_revenue.json", "w", encoding="utf-8") as f:
    json.dump(revenue_dict, f, indent=2)

pivot = status_pivot(df)
pivot.to_csv("status_pivot_table.csv")

ratios = item_refund_ratios(df)
ratios.to_csv("item_refund_ratios.csv")

print(pivot.round(2))
print(ratios.round(3))