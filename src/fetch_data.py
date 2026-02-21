import GEOparse
import pandas as pd
from pathlib import Path

def download_geo(gse_id: str, out_dir= "../data"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gse = GEOparse.get_GEO(geo=gse_id, destdir = out_dir, annotate_gpl=True)


    # Build expression matrix (samples as columns, genes as rows)
    dfs = []
    for gsm_name, gsm in gse.gsms.items():
        s = gsm.table
        s = s.rename(columns={s.columns[0]: "ID_REF", s.columns[-1]: gsm_name})
        dfs.append(s[["ID_REF", gsm_name]])
    expr = dfs[0]
    for d in dfs[1:]:
        expr = expr.merge(d, on="ID_REF", how= "inner")
    expr.set_index("ID_REF", inplace=True)

    # Sample metadata
    meta = []
    for gsm_name, gsm in gse.gsms.items():
        info = {"sample": gsm_name}
        for k, v in gsm.metadata.items():
            # Flatten first value where present
            if isinstance(v, list) and v:
                info[k] = v[0]
        # Often phenotype sits in characteristics_ch1 (or similar)
        ch = gsm.metadata.get("characteristics_ch1", [])
        for i, line, in enumerate (ch):
            key_val = line.split(":")
            if len(key_val) == 2:
                info[f"char{i}+{key_val[0].strip()}"] = key_val[1].strip()
            else:
                info[f"char{i}"] = line
        meta.append(info)
    meta = pd.DataFrame(meta).set_index("sample")

    expr.to_csv(Path(out_dir) / f"{gse_id}_expression.csv")
    meta.to_csv(Path(out_dir) / f"{gse_id}_metadata.csv")
    print(f"Saved: {gse_id}_expression.csv and {gse_id}_metadata.csv")
    return expr, meta

if __name__ == "__main__":
    download_geo("GSE10245")