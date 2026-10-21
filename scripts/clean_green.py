import sys, os, re, json, gc
import pandas as pd
import pyarrow.parquet as pq

# usage: python clean_green.py "D:\taxi\raw" "D:\taxi\clean_out\new_green"
raw_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
manifest_path = os.path.join(raw_dir, "_processed_green.json")   # separate from yellow

SAMPLE_SIZE = 100_000
SAMPLE_SEED = 42

KEEP = [
    "vendorid", "lpep_pickup_datetime", "lpep_dropoff_datetime",
    "store_and_fwd_flag", "ratecodeid", "pulocationid", "dolocationid",
    "passenger_count", "trip_distance", "fare_amount", "extra", "mta_tax",
    "tip_amount", "tolls_amount", "ehail_fee", "improvement_surcharge",
    "total_amount", "payment_type", "trip_type", "congestion_surcharge",
    "cbd_congestion_fee",
]

def signature(path):
    st = os.stat(path)
    return f"{st.st_size}-{int(st.st_mtime)}"

def read_sampled(src, target):
    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    wanted = [n for n in pf.schema_arrow.names if n.lower() in KEEP]
    frac = min(1.0, target / total) if total else 1.0
    parts = []
    for batch in pf.iter_batches(batch_size=250_000, columns=wanted):
        b = batch.to_pandas()
        if frac < 1.0:
            b = b.sample(frac=frac, random_state=SAMPLE_SEED)
        parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    return df

def clean(src):
    df = read_sampled(src, int(SAMPLE_SIZE * 1.1))

    for c in KEEP:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[KEEP]

    df = df.dropna(subset=["lpep_pickup_datetime", "lpep_dropoff_datetime",
                           "pulocationid", "dolocationid"])
    df = df[df["lpep_dropoff_datetime"] > df["lpep_pickup_datetime"]]

    m = re.search(r"(\d{4})-(\d{2})", os.path.basename(src))
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        p = df["lpep_pickup_datetime"]
        df = df[(p.dt.year == y) & (p.dt.month == mo)]

    for c in ["trip_distance", "fare_amount", "total_amount", "ehail_fee"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 200)]
    df = df[(df["fare_amount"] > 0) & (df["total_amount"] > 0)]

    df["passenger_count"] = (pd.to_numeric(df["passenger_count"], errors="coerce")
                             .fillna(1).clip(1, 6).astype(int))
    df["ratecodeid"] = (pd.to_numeric(df["ratecodeid"], errors="coerce")
                        .fillna(99).astype(int))
    df["trip_type"] = (pd.to_numeric(df["trip_type"], errors="coerce")
                       .fillna(1).astype(int))             # 1 = street-hail
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna("N")

    df["trip_minutes"] = ((df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"])
                          .dt.total_seconds() / 60).round(2)
    df = df[df["trip_minutes"].between(1, 360)]

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)

    df["source_file"] = os.path.basename(src)
    return df

manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
next_id = manifest.get("_next_id", 1)

all_files = []
for root, _, names in os.walk(raw_dir):
    for n in names:
        if n.lower().startswith("green") and n.lower().endswith(".parquet"):  # green only
            full = os.path.join(root, n)
            all_files.append((n, full, os.path.relpath(full, raw_dir)))
all_files.sort(key=lambda x: x[0])

new = [(n, full, rel) for n, full, rel in all_files
       if manifest.get(rel) != signature(full)]

if not new:
    print("No new parquet files.")
    sys.exit(0)

failed = 0
for name, src, rel in new:
    try:
        df = clean(src)
        df = df.sort_values(["lpep_pickup_datetime", "lpep_dropoff_datetime",
                             "pulocationid", "dolocationid"]).reset_index(drop=True)
        df.insert(0, "trip_id", range(next_id, next_id + len(df)))

        out = os.path.join(out_dir, name.replace(".parquet", ".csv"))
        df.to_csv(out, index=False, date_format="%Y-%m-%d %H:%M:%S")

        next_id += len(df)
        manifest[rel] = signature(src)
        manifest["_next_id"] = next_id
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        print(f"OK  {rel}: {len(df)} rows, ids up to {next_id - 1}")
        del df
        gc.collect()
    except Exception as e:
        failed += 1
        print(f"ERR {rel}: {e}", file=sys.stderr)

sys.exit(1 if failed else 0)