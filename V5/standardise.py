import duckdb
import pandas as pd


def standardise_bond_data(
    train_src="master_panel_train.parquet",
    test_src="master_panel_test.parquet",
    train_out="master_panel_train_std.parquet",
    test_out="master_panel_test_std.parquet",
    stats_out="standardization_stats.parquet",
):
    """
    Tier 1 (symmetric, mean-median gap < 0.1 std) -> plain z-score
        pr, prc_vw_par, ytm, LOCF_Leverage, interest_coverage_log,
        profitability, ret, vwretd, T10Y2Y

    Tier 2/3 ( skew, outliers pulling mean away from median) -> robust
    scaling: (x - median) / IQR, not distorted by extreme values the way
    mean/std are
        trade_count, days_since_last_trade, dvolume, VIXCLS, DFF, bond_age,
        mod_dur, BAMLH0A0HYM2, bid_ask_spread_filled, convexity,
        bond_maturity, accrued_since_last_trade, intraday_range_pct

    Magnitude/size variables -> log-transform first (converts multiplicative
    scale relationships into additive ones - THEN z-score
        bond_amt_outstanding, market_cap

    Identifiers, credit_spread (target), and binary/one-hot columns are
    left untouched throughout.
    """
    schema = duckdb.query(f"DESCRIBE SELECT * FROM '{train_src}'").df()['column_name'].tolist()

    identifiers = ['cusip', 'permno', 'permco', 'gvkey', 'date']
    target = ['credit_spread']

    binary_cols = ['144a_clean', 'call_clean', 'bid_ask_has_quote', 'interest_coverage_had_data']
    binary_cols += [c for c in schema if c.startswith('ff30_industry_')]
    binary_cols += [c for c in schema if c.startswith('spc_rat_') or c.startswith('mdc_rat_')]

    # days_since_last_trade was sentinel-filled with -1 for a bond's first
    # observation (real gap unknown).

    tier1_cols = ['pr', 'prc_vw_par', 'ytm', 'LOCF_Leverage', 'interest_coverage_log',
                  'profitability', 'ret', 'vwretd', 'T10Y2Y']

    tier23_cols = ['trade_count', 'days_since_last_trade', 'dvolume', 'VIXCLS', 'DFF',
                   'bond_age', 'mod_dur', 'BAMLH0A0HYM2', 'bid_ask_spread_filled',
                   'convexity', 'bond_maturity', 'accrued_since_last_trade', 'intraday_range_pct']

    log_then_zscore_cols = ['bond_amt_outstanding', 'market_cap']


    all_continuous = tier1_cols + tier23_cols + log_then_zscore_cols
    expected_continuous = [c for c in schema if c not in identifiers + target + binary_cols]
    missing = set(expected_continuous) - set(all_continuous)
    extra = set(all_continuous) - set(expected_continuous)
    if missing:
        raise ValueError(f"Columns not assigned to any tier: {missing}")
    if extra:
        raise ValueError(f"Columns assigned but not in schema, or duplicated: {extra}")

    # --- Compute stats from TRAIN only ---
    stats_query = f"""
        SELECT
            {", ".join([f'AVG("{c}") AS "{c}_mean", STDDEV("{c}") AS "{c}_std"' for c in tier1_cols])},
            {", ".join([f'approx_quantile("{c}", 0.5) AS "{c}_median", (approx_quantile("{c}", 0.75) - approx_quantile("{c}", 0.25)) AS "{c}_iqr"' for c in tier23_cols])},
            {", ".join([f'AVG(LN(1 + ABS("{c}"))) AS "{c}_log_mean", STDDEV(LN(1 + ABS("{c}"))) AS "{c}_log_std"' for c in log_then_zscore_cols])}
        FROM '{train_src}'
    """
    stats = duckdb.query(stats_query).df().iloc[0]

    stats_rows = []
    for c in tier1_cols:
        stats_rows.append({'column': c, 'method': 'zscore', 'center': stats[f"{c}_mean"], 'scale': stats[f"{c}_std"]})
    for c in tier23_cols:
        stats_rows.append({'column': c, 'method': 'robust', 'center': stats[f"{c}_median"], 'scale': stats[f"{c}_iqr"]})
    for c in log_then_zscore_cols:
        stats_rows.append({'column': c, 'method': 'log_then_zscore', 'center': stats[f"{c}_log_mean"], 'scale': stats[f"{c}_log_std"]})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_parquet(stats_out)
    print("Saved", stats_out)
    print(stats_df)

    # Apply to both train and test using train's stats
    def _build_query(src, out):
        exprs = []
        for c in tier1_cols:
            exprs.append(f'(CAST("{c}" AS DOUBLE) - {stats[f"{c}_mean"]}) / {stats[f"{c}_std"]} AS "{c}"')
        for c in tier23_cols:
            exprs.append(f'(CAST("{c}" AS DOUBLE) - {stats[f"{c}_median"]}) / {stats[f"{c}_iqr"]} AS "{c}"')
        # Flag added here, BEFORE days_since_last_trade's -1 sentinel gets
        # scaled into an arbitrary number indistinguishable from a real gap
        exprs.append('CASE WHEN "days_since_last_trade" = -1 THEN 0 ELSE 1 END AS "has_prior_trade"')
        for c in log_then_zscore_cols:
            exprs.append(
                f'(LN(1 + ABS(CAST("{c}" AS DOUBLE))) - {stats[f"{c}_log_mean"]}) / {stats[f"{c}_log_std"]} AS "{c}"'
            )
        unchanged_cols = [f'"{c}"' for c in identifiers + target + binary_cols]
        return f"""
            COPY (
                SELECT {", ".join(unchanged_cols)}, {", ".join(exprs)}
                FROM '{src}'
            ) TO '{out}' (FORMAT PARQUET)
        """

    duckdb.query(_build_query(train_src, train_out))
    duckdb.query(_build_query(test_src, test_out))
    print(f"\nSaved {train_out} and {test_out}")

    return {
        "train_path": train_out,
        "test_path": test_out,
        "stats_path": stats_out,
        "identifiers": identifiers,
        "target": target,
        "binary_cols": binary_cols + ["has_prior_trade"],
        "tier1_cols": tier1_cols,
        "tier23_cols": tier23_cols,
        "log_then_zscore_cols": log_then_zscore_cols,
        "continuous_cols": all_continuous,
    }
