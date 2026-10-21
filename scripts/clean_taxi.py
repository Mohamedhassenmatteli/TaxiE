import sys, os, re, json
import pandas as pd
import pyarrow.parquet as pq

# ---------- arguments ----------
# usage: python clean_taxi.py "D:\taxi\raw" "D:\taxi\clean_out\new"
raw_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
manifest_path = os.path.join(raw_dir, "_processed.json")

# ---------- settings ----------
SAMPLE_SIZE = 100_000     # max rows kept per monthly file (change to 100_000 to go lighter)
SAMPLE_SEED = 42             # same seed = same sample on reruns

KEEP = [
    "vendorid", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "ratecodeid", "store_and_fwd_flag",
    "pulocationid", "dolocationid", "payment_type", "fare_amount", "extra",
    "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
    "congestion_surcharge", "airport_fee", "cbd_congestion_fee", "total_amount",
]

def signature(path):
    """Changes if the file is replaced/modified, so it gets reprocessed."""
    st = os.stat(path)
    return f"{st.st_size}-{int(st.st_mtime)}"

def clean(src):
    # read only the needed columns (column name case differs across years)
    names = pq.read_schema(src).names
    wanted = [n for n in names if n.lower() in KEEP]
    df = pd.read_parquet(src, columns=wanted)
    df.columns = [c.lower() for c in df.columns]

    # sample early (10% extra to leave room for rows removed by cleaning)
    pre = int(SAMPLE_SIZE * 1.1)
    if len(df) > pre:
        df = df.sample(n=pre, random_state=SAMPLE_SEED)

    # add missing columns as NULL, fixed column order
    for c in KEEP:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[KEEP]

    # required fields
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime",
                           "pulocationid", "dolocationid"])
    df = df[df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]]

    # keep only rows matching the file's year-month (TLC files contain stray dates)
    m = re.search(r"(\d{4})-(\d{2})", os.path.basename(src))
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        p = df["tpep_pickup_datetime"]
        df = df[(p.dt.year == y) & (p.dt.month == mo)]

    # numeric checks
    for c in ["trip_distance", "fare_amount", "total_amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 200)]
    df = df[(df["fare_amount"] > 0) & (df["total_amount"] > 0)]

    # defaults / types
    df["passenger_count"] = (pd.to_numeric(df["passenger_count"], errors="coerce")
                             .fillna(1).clip(1, 6).astype(int))
    df["ratecodeid"] = (pd.to_numeric(df["ratecodeid"], errors="coerce")
                        .fillna(99).astype(int))          # 99 = unknown
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna("N")

    # derived column + duration filter
    df["trip_minutes"] = ((df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
                          .dt.total_seconds() / 60).round(2)
    df = df[df["trip_minutes"].between(1, 360)]

    # final cap
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)

    df["source_file"] = os.path.basename(src)
    return df

# ---------- load manifest + persistent ID counter ----------
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
next_id = manifest.get("_next_id", 1)

# ---------- find all parquet files in every subfolder ----------
all_files = []
for root, _, names in os.walk(raw_dir):
    for n in names:
        if n.lower().endswith(".parquet"):
            full = os.path.join(root, n)
            all_files.append((n, full, os.path.relpath(full, raw_dir)))
all_files.sort(key=lambda x: x[0])      # chronological by file name

new = [(n, full, rel) for n, full, rel in all_files
       if manifest.get(rel) != signature(full)]

if not new:
    print("No new parquet files.")
    sys.exit(0)

# ---------- process ----------
failed = 0
for name, src, rel in new:
    try:
        df = clean(src)

        # deterministic order, then generate the primary key
        df = df.sort_values(["tpep_pickup_datetime", "tpep_dropoff_datetime",
                             "pulocationid", "dolocationid"]).reset_index(drop=True)
        df.insert(0, "trip_id", range(next_id, next_id + len(df)))

        out = os.path.join(out_dir, name.replace(".parquet", ".csv"))
        df.to_csv(out, index=False, date_format="%Y-%m-%d %H:%M:%S")

        next_id += len(df)
        manifest[rel] = signature(src)
        manifest["_next_id"] = next_id          # saved only after the file succeeds
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        print(f"OK  {rel}: {len(df)} rows, ids up to {next_id - 1}")
    except Exception as e:
        failed += 1
        print(f"ERR {rel}: {e}", file=sys.stderr)

sys.exit(1 if failed else 0)