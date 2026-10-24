"""Manufacturing peak forecasting, extracted from the competition notebook.

Scientific functions and fixed parameters retain the source behavior.
See docs/provenance.md for the extraction and result provenance.
"""
from pathlib import Path
import importlib
import json
import os
import time
import warnings
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PEAK_DATA_DIR", str(PROJECT_ROOT / "data")))
ROOT = DATA_DIR
IN_XLSX = DATA_DIR / "okm_전처리완료_2021.xlsx"
N_JOBS = 1



def optional_import(name):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None

lgb = optional_import("lightgbm")
sk_ensemble = optional_import("sklearn.ensemble")
RandomForestRegressor = None if sk_ensemble is None else sk_ensemble.RandomForestRegressor
joblib_module = optional_import("joblib")
Parallel = None if joblib_module is None else joblib_module.Parallel
delayed = None if joblib_module is None else joblib_module.delayed

def require_training_dependencies():
    missing = []
    if lgb is None: missing.append("lightgbm")
    if RandomForestRegressor is None: missing.append("scikit-learn")
    if missing:
        raise ImportError("모델 실행 라이브러리 설치 필요: " + ", ".join(missing))



TARGET = "최대15분"
SEL_START, SEL_END = "2021-07-05", "2021-08-30"
FINAL_START, DATA_END = "2021-08-30", "2021-09-14"
PEAK_TH, EXT_TH = 180.0, 185.0
SEEDS = [42, 7, 2024]
CONFIRM_SEEDS = [1, 123, 999]
ADD_MAE_GAIN, ADD_PEAK_TOL = 0.20, 0.50
STATE_LAG_DAYS = 5
BANNED = ["15분", "30분", "45분", "60분", "평균", "최대15분", "최대15분_래그용",
          "공장인원", "순서훼손", "전력결측", "시간단위_학습사용", "날짜", "date", "ts"]
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1)
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=3, n_jobs=1)
FEAT_A = ["시간", "요일", "월", "생산량", "계획_전1h", "계획_후1h", "일계획합", "계획가동상태",
          "기온", "일최고기온", "습도", "풍속", "시간당강수량", "가동상태래그"]
FEAT_B = FEAT_A + ["전력_1h전", "전력_2h전"]
B_ONLY = ["전력_1h전", "전력_2h전"]
POWER_COLS = ["15분", "30분", "45분", "60분", "평균", "최대15분", "최대15분_래그용"]

# ===== 피크 중심 고도화 설정 =====
PEAK_MAE_GUARDRAIL_KW = 1.0
PEAK_MIN_GAIN_KW = 0.10
PEAK_SAMPLE_WEIGHT = 2.0
EXTREME_SAMPLE_WEIGHT = 3.0

PEAK_MODEL_VARIANTS = [
    "LightGBM",
    "LGB_PeakW",
    "LGB_Q60",
    "LGB_Q65",
    "LGB_PeakW_Q60",
    "LGB_PeakW_Q65",
]

RUN_PEAK_TUNING = False
RUN_PEAK_FEATURE_REFINEMENT = False

# selection 구간에서 확정한 뒤 동결
FINAL_MODEL_A = "LGB_Q65"
FINAL_MODEL_B = "LGB_PeakW_Q65"


def load_data(path=IN_XLSX):
    if not Path(path).exists():
        raise FileNotFoundError(f"입력 데이터 없음: {path}")
    df = pd.read_excel(path, sheet_name="데이터")
    df["date"] = pd.to_datetime(df["날짜"].astype(int).astype(str), format="%Y%m%d")
    df = df.sort_values(["date", "시간"]).reset_index(drop=True)
    df["ts"] = df["date"] + pd.to_timedelta(df["시간"], unit="h")
    return df

def find_copy_days(df):
    """하루 96칸 전력이 다른 날과 완전히 같은 날 = 복사본 (가장 이른 날만 원본). 학습에서 제외."""
    q = ["15분", "30분", "45분", "60분"]
    groups = {}
    for day, g in df.groupby("date"):
        if len(g) == 24 and g[q].notna().all().all():
            groups.setdefault(tuple(g[q].values.flatten()), []).append(day)
    return {d for days in groups.values() if len(days) > 1 for d in sorted(days)[1:]}

def add_day_info(df):
    """계획가동상태: 1=가동(계획>0), 0=비가동(하루 내내 0), NaN=미상(하루 내내 결측, 7/13·7/15)"""
    day = df.groupby("date").agg(일계획합=("생산량", lambda s: s.sum(min_count=1)),
                                 일최고기온=("기온", "max"))
    day["계획가동상태"] = np.where(day["일계획합"].isna(), np.nan, (day["일계획합"] > 0).astype(float))
    gap, prev = [], None
    for d, r in day.iterrows():
        gap.append((d - prev).days if prev is not None else 1)
        if r["계획가동상태"] == 1:
            prev = d
    day["직전가동간격"] = gap
    return df.merge(day, left_on="date", right_index=True, how="left")

def add_state_lag(df):
    """가동상태래그: 대상일과 같은 계획가동상태였던 과거 최근 5일의 같은 시각 전력 평균.
    A 발행(전날 23:00) 기준으로 안전하게: 23시 칸은 전날(d-1)을 비교일에서 뺌 (전날 23~24시는 아직 모름).
    미상 날은 비교일로 쓰지 않고, 미상 날 자체의 래그는 NaN."""
    piv = df.pivot_table(index="date", columns="시간", values="최대15분_래그용")
    state = df.groupby("date")["계획가동상태"].first()
    days = list(piv.index)
    rows = []
    for i, d in enumerate(days):
        if pd.isna(state[d]):
            rows.append(pd.Series(np.nan, index=piv.columns, name=d)); continue
        pool = [p for p in days[:i] if state[p] == state[d] and piv.loc[p].notna().all()]
        v = piv.loc[pool[-STATE_LAG_DAYS:]].mean() if pool else pd.Series(np.nan, index=piv.columns)
        pool23 = [p for p in pool if p != d - pd.Timedelta(days=1)]
        v[23] = piv.loc[pool23[-STATE_LAG_DAYS:], 23].mean() if pool23 else np.nan
        rows.append(v.rename(d))
    lag = pd.DataFrame(rows)
    lag.columns = [int(c) for c in lag.columns]
    lag = lag.rename_axis("date").reset_index().melt(id_vars="date", var_name="시간", value_name="가동상태래그")
    lag["시간"] = lag["시간"].astype(int)
    return df.merge(lag, on=["date", "시간"], how="left")

def add_base_features(df):
    df = df.copy()
    df["요일"] = df["date"].dt.dayofweek
    df["월"] = df["date"].dt.month
    df["계획_전1h"] = df["생산량"].shift(1)
    df["계획_후1h"] = df["생산량"].shift(-1)
    df.loc[df["시간"] == 23, "계획_후1h"] = np.nan          # 다음 날 0시 계획 = A 발행 시점에 모름
    df = add_state_lag(df)
    df["전력_1h전"] = df["최대15분_래그용"].shift(1)        # B 전용
    df["전력_2h전"] = df["최대15분_래그용"].shift(2)        # B 전용
    return df

def prepare(add_candidates=None, path=IN_XLSX):
    """전체 전처리 → (전체 df, 사용 가능 행 usable, 복사본 날짜 집합).
    add_candidates(df) -> df : 새 피처 컬럼을 추가해 돌려주는 함수 (전체 행 기준으로 계산할 것)"""
    df = load_data(path)
    copy_days = find_copy_days(df)
    df = add_base_features(add_day_info(df))
    if add_candidates is not None:
        n0 = set(df.columns)
        df = add_candidates(df.copy())
        bad = [c for c in set(df.columns) - n0 if c in BANNED]
        if bad:
            raise ValueError(f"금지 컬럼 이름 사용: {bad}")
    usable = df[df["시간단위_학습사용"] == 1].copy()
    return df, usable, copy_days

DUP = {"target_hour": "시간", "target_dow": "요일", "target_month": "월", "plan_target": "생산량",
       "plan_offset_m1": "계획_전1h", "plan_offset_p1": "계획_후1h", "plan_day_sum": "일계획합",
       "horizon_hours": "시간(+1)"}

def add_candidates(df):
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    assert (df["ts"].diff().dropna() == pd.Timedelta(hours=1)).all(), "시간 행이 연속이어야 함"
    P = df["최대15분_래그용"].astype(float)
    Q = df["생산량"].astype(float)
    W = {"temp": df["기온"], "wind": df["풍속"], "humidity": df["습도"], "rain": df["시간당강수량"]}
    date = df["date"]
    day0 = df.groupby("date").cumcount().eq(0)
    pos = np.arange(n)
    first_pos = pd.Series(np.where(day0, pos, np.nan)).ffill().astype(int).values
    a_pos = first_pos - 1                        # 전날 23시 행 위치 (컷오프 C_A)

    def take(S, h):
        S = np.asarray(S, dtype=float)
        if h == "B":
            return S
        out = np.full(n, np.nan)
        ok = a_pos >= 0
        out[ok] = S[a_pos[ok]]
        return out

    new = {}

    def put(name, S=None, A=None, B=None):
        new[f"A_{name}"] = take(S, "A") if A is None else np.asarray(A, float)
        new[f"B_{name}"] = take(S, "B") if B is None else np.asarray(B, float)

    def same(name, v):                          # 컷오프와 무관 (대상 시각 T 기준)
        v = np.asarray(v, dtype=float)
        new[f"A_{name}"] = v; new[f"B_{name}"] = v

    # ---------- plan ----------
    same_day = lambda k: (date.shift(-k) == date)
    same("plan_target_log", np.log1p(Q))
    same("plan_offset_m2", Q.shift(2))
    same("plan_offset_p2", Q.shift(-2).where(same_day(2)))
    Qfwd = lambda k: Q.shift(-k).where(same_day(k))
    prev3 = pd.concat([Q.shift(k) for k in (1, 2, 3)], axis=1)
    next3 = pd.concat([Qfwd(k) for k in (0, 1, 2)], axis=1)
    prev8 = pd.concat([Q.shift(k) for k in range(1, 9)], axis=1)
    for nm, M in [("prev3", prev3), ("next3", next3), ("prev8", prev8)]:
        same(f"plan_{nm}_sum", M.sum(axis=1, min_count=1))
        same(f"plan_{nm}_mean", M.mean(axis=1))
        same(f"plan_{nm}_max", M.max(axis=1))
    g = Q.groupby(date)
    same("plan_day_mean", g.transform("mean"))
    same("plan_day_max", g.transform("max"))
    same("plan_day_std", g.transform("std"))
    # lead = [C, T+1h)
    cumQ = Q.fillna(0).groupby(date).cumsum()
    q_prev23 = take(Q, "A")                     # 전날 23시 계획
    lead_sum_A = cumQ.values + np.nan_to_num(q_prev23)
    cnt = df.groupby("date").cumcount().values + 2
    cummax = Q.groupby(date).cummax().values
    put("plan_lead_sum", A=lead_sum_A, B=Q.values)
    put("plan_lead_mean", A=lead_sum_A / cnt, B=Q.values)
    put("plan_lead_max", A=np.fmax(cummax, q_prev23), B=Q.values)
    for k in (1, 2, 7, 14):
        same(f"plan_same_hour_d{k}", Q.shift(24 * k))
    same("plan_diff_d1", Q - Q.shift(24))
    same("plan_diff_d7", Q - Q.shift(168))
    same("plan_prev_hour_diff", Q - Q.shift(1))
    Qs = Q.shift(1)
    put("plan_hist_24_mean", Qs.rolling(24, min_periods=1).mean())
    put("plan_hist_24_std", Qs.rolling(24, min_periods=2).std())
    put("plan_hist_168_mean", Qs.rolling(168, min_periods=1).mean())
    put("plan_hist_168_std", Qs.rolling(168, min_periods=2).std())
    h24 = new["A_plan_hist_24_mean"], new["B_plan_hist_24_mean"]
    new["A_plan_vs_hist24"] = np.where(h24[0] > 0, Q / np.where(h24[0] > 0, h24[0], 1), np.nan)
    new["B_plan_vs_hist24"] = np.where(h24[1] > 0, Q / np.where(h24[1] > 0, h24[1], 1), np.nan)

    # ---------- power (컷오프 기준) ----------
    Ps = P.shift(1)                              # S[t] 에서 t-1 행까지
    for k in (1, 2, 3, 6, 8, 12, 24, 48, 72, 168, 336, 672):
        put(f"power_cutoff_lag{k}", P.shift(k))
    for k in (2, 3, 7, 14, 21, 28):
        same(f"power_same_hour_d{k}", P.shift(24 * k))
    new["B_power_same_hour_d1"] = P.shift(24).values.astype(float)
    for w in (3, 8, 24, 72, 168):
        r = Ps.rolling(w, min_periods=1)
        put(f"power_roll{w}_mean", r.mean())
        put(f"power_roll{w}_std", Ps.rolling(w, min_periods=2).std())
        put(f"power_roll{w}_min", r.min())
        put(f"power_roll{w}_max", r.max())
        put(f"power_roll{w}_median", r.median())
    for sp in (6, 24, 72, 168):
        put(f"power_ewma{sp}", Ps.ewm(span=sp, adjust=False, ignore_na=True).mean())
    put("power_diff1_2", P.shift(1) - P.shift(2))
    put("power_diff1_24", P.shift(1) - P.shift(24))
    put("power_diff24_48", P.shift(24) - P.shift(48))
    put("power_trend24_168", Ps.rolling(24, min_periods=1).mean() - Ps.rolling(168, min_periods=1).mean())
    m24 = Ps.rolling(24, min_periods=1).mean()
    put("power_ratio1_24", P.shift(1) / m24.where(m24 > 0))
    wk = pd.concat([P.shift(24 * k) for k in (7, 14, 21, 28)], axis=1)
    same("power_weekhour4_mean", wk.mean(axis=1))
    same("power_weekhour4_std", wk.std(axis=1))
    same("power_weekhour4_max", wk.max(axis=1))
    dmax = P.groupby(date).max()
    for nm, lagd in [("A", 2), ("B", 1)]:        # 완료된 날: A는 전전날까지, B는 전날까지
        base = dmax.shift(lagd)
        for st in ("mean", "std", "max"):
            v = getattr(base.rolling(7, min_periods=3), st)()
            new[f"{nm}_power_daily_peak7_{st}"] = date.map(v).values.astype(float)
    put("power_roll168_q90", Ps.rolling(168, min_periods=24).quantile(0.9))


    # ---------- peak-specific history ----------
    same("hour_sin", np.sin(2 * np.pi * df["시간"] / 24))
    same("hour_cos", np.cos(2 * np.pi * df["시간"] / 24))

    # A: 0~22시는 전일 동일시각, 23시는 발행시점상 전날 23시 실측 미완료
    a_same_d1_safe = P.shift(24).copy()
    a_same_d1_safe = a_same_d1_safe.where(df["시간"] != 23, P.shift(48))
    new["A_power_same_hour_d1_safe"] = a_same_d1_safe.values.astype(float)

    # 최근 고부하 regime
    peak_hist = P.shift(1).ge(PEAK_TH).astype(float).where(P.shift(1).notna())
    ext_hist = P.shift(1).ge(EXT_TH).astype(float).where(P.shift(1).notna())
    for w in (24, 72, 168):
        put(f"peak_rate{w}", peak_hist.rolling(w, min_periods=max(6, w // 4)).mean())
    put("extreme_rate168", ext_hist.rolling(168, min_periods=24).mean())

    # 최근 4주 같은 요일·같은 시각의 분포
    same("power_weekhour4_q75", wk.quantile(0.75, axis=1))
    same("power_weekhour4_q90", wk.quantile(0.90, axis=1))
    same("power_weekhour4_peak_rate", (wk >= PEAK_TH).mean(axis=1))

    # B: 직전 1시간의 15분 내부 shape
    q15 = df["15분"].shift(1).astype(float)
    q30 = df["30분"].shift(1).astype(float)
    q45 = df["45분"].shift(1).astype(float)
    q60 = df["60분"].shift(1).astype(float)
    new["B_prev_q15"] = q15.values
    new["B_prev_q30"] = q30.values
    new["B_prev_q45"] = q45.values
    new["B_prev_q60"] = q60.values
    qmat = pd.concat([q15, q30, q45, q60], axis=1)
    new["B_prev_hour_range"] = (qmat.max(axis=1) - qmat.min(axis=1)).values
    new["B_prev_hour_slope"] = (q60 - q15).values
    new["B_prev_hour_last"] = q60.values

    # ---------- calendar ----------
    d = date.dt
    same("target_weekend", (d.dayofweek >= 5).astype(float))
    same("target_day", d.day)
    same("target_doy", d.dayofyear)
    same("target_season", (d.month % 12) // 3)
    same("month_sin", np.sin(2 * np.pi * d.month / 12)); same("month_cos", np.cos(2 * np.pi * d.month / 12))
    same("dow_sin", np.sin(2 * np.pi * d.dayofweek / 7)); same("dow_cos", np.cos(2 * np.pi * d.dayofweek / 7))
    same("month_end", d.is_month_end.astype(float))
    new["A_cutoff_dow"] = ((d.dayofweek - 1) % 7).values.astype(float)
    same("tariff_month", df["전기요금(계절)"])

    # ---------- operation ----------
    act = (Q > 0).astype(float).where(Q.notna())
    same("plan_active", act)
    same("plan_day_active_hours", act.groupby(date).transform("sum"))
    lead_act_A = act.fillna(0).groupby(date).cumsum().values + np.nan_to_num(take(act, "A"))
    put("plan_lead_active_hours", A=lead_act_A, B=act.values)
    same("plan_start", ((Q > 0) & (Q.shift(1) == 0)).astype(float).where(Q.notna()))
    same("plan_stop", ((Q == 0) & (Q.shift(1) > 0)).astype(float).where(Q.notna()))
    run = np.zeros(n); c = 0
    a = act.values
    for i in range(n):                           # T 직전 연속 가동 시간 (최대 24)
        run[i] = c
        c = min(c + 1, 24) if a[i] == 1 else 0
    same("plan_prior_active_run", run)

    # ---------- weather ----------
    for k, S in W.items():
        put(f"{k}_last", S.shift(1))
        put(f"{k}_mean24", S.shift(1).rolling(24, min_periods=1).mean())
        same(f"{k}_samehour", S.shift(48))
    T = W["temp"]
    put("temp_trend24", T.shift(1) - T.shift(25))
    for h in ("A", "B"):
        tm = new[f"{h}_temp_mean24"]
        new[f"{h}_cooling_degree_proxy"] = np.maximum(tm - 24, 0)
        new[f"{h}_heating_degree_proxy"] = np.maximum(18 - tm, 0)
        new[f"{h}_plan_x_temp"] = Q.values * tm
        new[f"{h}_plan_x_humidity"] = Q.values * new[f"{h}_humidity_mean24"] / 100
        new[f"{h}_plan_x_cooling"] = Q.values * new[f"{h}_cooling_degree_proxy"]
        new[f"{h}_plan_x_recentpower"] = Q.values * new[f"{h}_power_roll24_mean"]
    same("plan_x_weekend", Q * (d.dayofweek >= 5))
    same("plan_x_dow_sin", Q * np.sin(2 * np.pi * d.dayofweek / 7))

    # ---------- relation / quality ----------
    put("power_plan_corr168", Ps.rolling(168, min_periods=24).corr(Qs))
    on = (Qs > 0).astype(float)
    off = (Qs == 0).astype(float)
    put("power_active_mean168", (Ps * on).rolling(168, min_periods=1).sum() / on.rolling(168, min_periods=1).sum().replace(0, np.nan))
    put("power_idle_mean168", (Ps * off).rolling(168, min_periods=1).sum() / off.rolling(168, min_periods=1).sum().replace(0, np.nan))
    miss = (df["시간단위_학습사용"] != 1).astype(float).shift(1)
    put("power_missing_frac24", miss.rolling(24, min_periods=1).mean())
    put("power_missing_frac168", miss.rolling(168, min_periods=1).mean())

    out = pd.DataFrame(new, index=df.index).replace([np.inf, -np.inf], np.nan)
    return pd.concat([df, out], axis=1)

def cand_lists(df_cols):
    A = [c for c in df_cols if c.startswith("A_")]
    B = [c for c in df_cols if c.startswith("B_")]
    return A, B


def _lgb_variant(model):
    return {
        "LightGBM": (False, None),
        "LGB_PeakW": (True, None),
        "LGB_Q60": (False, 0.60),
        "LGB_Q65": (False, 0.65),
        "LGB_PeakW_Q60": (True, 0.60),
        "LGB_PeakW_Q65": (True, 0.65),
    }.get(model)

def fit_predict(model, tr, te, feats, seed):
    require_training_dependencies()
    bad = [f for f in feats if f in BANNED]
    if bad:
        raise ValueError(f"금지 피처 사용: {bad}")

    X, y = tr[feats], tr[TARGET]
    variant = _lgb_variant(model)

    if variant is not None:
        use_peak_weight, alpha = variant
        params = dict(LGB_PARAMS)
        if alpha is not None:
            params.update(objective="quantile", alpha=alpha)

        sample_weight = None
        if use_peak_weight:
            sample_weight = np.ones(len(y), dtype=float)
            sample_weight[y >= PEAK_TH] = PEAK_SAMPLE_WEIGHT
            sample_weight[y >= EXT_TH] = EXTREME_SAMPLE_WEIGHT

        m = lgb.LGBMRegressor(random_state=seed, **params)
        m.fit(X, y, sample_weight=sample_weight)
        return m.predict(te[feats])

    if model == "RandomForest":
        m = RandomForestRegressor(random_state=seed, **RF_PARAMS).fit(X.fillna(-1), y)
        return m.predict(te[feats].fillna(-1))

    raise ValueError(model)

def weeks(period="select"):
    end = pd.Timestamp(DATA_END) + pd.Timedelta(days=1)
    if period == "select":
        st = pd.date_range(SEL_START, pd.Timestamp(SEL_END) - pd.Timedelta(days=1), freq="W-MON")
        return [(x, x + pd.Timedelta(days=7)) for x in st]
    if period == "final":
        return [
            (pd.Timestamp("2021-08-30"), pd.Timestamp("2021-09-06")),
            (pd.Timestamp("2021-09-06"), end),
        ]
    raise ValueError(period)

def rolling(usable, copy_days, feats, horizon, model="LightGBM", seed=42, period="select"):
    if horizon == "A" and any(f in B_ONLY for f in feats):
        raise ValueError("시점 A에 B 전용 피처 사용 금지")

    out = []
    for ws, we in weeks(period):
        train_cutoff = ws - pd.Timedelta(hours=1) if horizon == "A" else ws

        tr = usable[
            (usable["ts"] < train_cutoff) &
            (~usable["date"].isin(copy_days))
        ]
        te = usable[(usable["ts"] >= ws) & (usable["ts"] < we)].copy()
        if te.empty:
            continue

        te["pred"] = fit_predict(model, tr, te, feats, seed)
        te["주시작"] = ws.date()
        out.append(te)

    if not out:
        raise ValueError("rolling 평가 결과가 비었습니다.")
    return pd.concat(out)

def metrics(g):
    e = g[TARGET] - g["pred"]
    ae = e.abs()
    pk = g[TARGET] >= PEAK_TH
    ex = g[TARGET] >= EXT_TH
    pred_pk = g["pred"] >= PEAK_TH
    under = e[pk].clip(lower=0)

    tp = int((pk & pred_pk).sum())
    fn = int((pk & ~pred_pk).sum())
    fp = int((~pk & pred_pk).sum())

    return pd.Series({
        "MAE": ae.mean(),
        "피크MAE": ae[pk].mean(),
        "과소예측피크": under.mean(),
        "과소예측P95": under.quantile(0.95) if len(under) else np.nan,
        "최대과소예측": under.max() if len(under) else np.nan,
        "PeakRecall": tp / (tp + fn) if (tp + fn) else np.nan,
        "PeakPrecision": tp / (tp + fp) if (tp + fp) else np.nan,
        "MissedPeak": fn,
        "극단MAE": ae[ex].mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "시간수": len(g),
        "피크시간수": int(pk.sum()),
    })

def _one(usable, copy_days, feats, horizon, model, seed, period):
    return metrics(rolling(usable, copy_days, feats, horizon, model, seed, period))

def evaluate(usable, copy_days, sets, horizon, model="LightGBM",
             seeds=None, period="select", n_jobs=N_JOBS):
    if period == "final":
        raise RuntimeError("최종 채점은 final_score()만 사용")
    seeds = seeds or SEEDS
    jobs = [(n, f, sd) for n, f in sets.items() for sd in seeds]
    res = (
        Parallel(n_jobs=n_jobs)(
            delayed(_one)(usable, copy_days, f, horizon, model, sd, period)
            for _, f, sd in jobs
        )
        if Parallel is not None else
        [_one(usable, copy_days, f, horizon, model, sd, period) for _, f, sd in jobs]
    )
    z = pd.DataFrame([r.rename(n) for (n, _, _), r in zip(jobs, res)])
    return z.groupby(level=0).mean()

def evaluate_peak_models(usable, copy_days, feats, horizon, seeds=None):
    rows = []
    for model in PEAK_MODEL_VARIANTS:
        r = evaluate(
            usable, copy_days, {"score": feats}, horizon,
            model=model, seeds=seeds or SEEDS, period="select"
        ).loc["score"]
        r.name = model
        rows.append(r)

    E = pd.DataFrame(rows)
    base_mae = E.loc["LightGBM", "MAE"]
    E["MAE_guardrail"] = E["MAE"] <= base_mae + PEAK_MAE_GUARDRAIL_KW
    E["선택순위"] = np.nan

    ok = E[E["MAE_guardrail"]].copy()
    order = ok.sort_values(
        ["피크MAE", "과소예측P95", "PeakRecall"],
        ascending=[True, True, False]
    ).index
    for rank, name in enumerate(order, 1):
        E.loc[name, "선택순위"] = rank

    return E.sort_values(["MAE_guardrail", "선택순위"], ascending=[False, True])

def compare_add_peak(usable, copy_days, base, cands, horizon, model, seeds=None):
    cands = [c for c in cands if c not in base]
    sets = {"기준": base, **{f"+{c}": base + [c] for c in cands}}
    E = evaluate(usable, copy_days, sets, horizon, model=model, seeds=seeds or SEEDS)
    b = E.loc["기준"]

    E["MAE변화"] = E["MAE"] - b["MAE"]
    E["피크MAE개선"] = b["피크MAE"] - E["피크MAE"]
    E["과소P95개선"] = b["과소예측P95"] - E["과소예측P95"]
    E["Recall개선"] = E["PeakRecall"] - b["PeakRecall"]

    guard = E["MAE"] <= b["MAE"] + PEAK_MAE_GUARDRAIL_KW
    gain = E["피크MAE개선"] >= PEAK_MIN_GAIN_KW
    E["판정"] = np.where(guard & gain, "피크채택후보", "보류")
    E.loc["기준", "판정"] = "-"
    return E.sort_values(
        ["피크MAE", "과소예측P95", "PeakRecall"],
        ascending=[True, True, False]
    )

def final_score(usable, copy_days, feats_a, feats_b, frozen=False, out_dir=None,
                model_a="LightGBM", model_b="LightGBM"):
    if not frozen:
        raise RuntimeError("selection 구간에서 모델·피처를 동결한 뒤 frozen=True로 실행")

    out_dir = Path(out_dir or ROOT / "최종결과")
    out_dir.mkdir(exist_ok=True)
    rows, preds = [], []

    for h, feats, chosen_model in [
        ("A", feats_a, model_a),
        ("B", feats_b, model_b),
    ]:
        model_list = []
        for model in [chosen_model, "LightGBM", "RandomForest"]:
            if model not in model_list:
                model_list.append(model)

        for model in model_list:
            g = rolling(usable, copy_days, feats, h, model, seed=42, period="final")
            rows.append(metrics(g).rename(f"{h}_{model}"))
            preds.append(
                g[["ts", "시간", TARGET, "pred"]].assign(시점=h, 모델=model)
            )

    S = pd.DataFrame(rows).round(3)
    S.to_csv(out_dir / "최종성능.csv", encoding="utf-8-sig")
    pd.concat(preds).to_csv(
        out_dir / "예측결과_테스트.csv", index=False, encoding="utf-8-sig"
    )
    return S


def compare_add(usable, copy_days, base, cands, horizon, model="LightGBM", seeds=None):
    """기준 세트에 후보를 1개씩 추가해 비교 → 판정표"""
    sets = {"기준": base, **{f"+{c}": base + [c] for c in cands if c not in base}}
    E = evaluate(usable, copy_days, sets, horizon, model, seeds)
    b = E.loc["기준"]
    E["MAE변화"] = E["MAE"] - b["MAE"]
    E["피크MAE변화"] = E["피크MAE"] - b["피크MAE"]
    E["판정"] = np.where((E["MAE변화"] <= -ADD_MAE_GAIN) & (E["피크MAE변화"] <= ADD_PEAK_TOL), "채택후보", "탈락")
    E.loc["기준", "판정"] = "-"
    return E.sort_values("MAE변화")

def compare_drop(usable, copy_days, full, horizon, model="LightGBM", seeds=None):
    """전체 세트에서 1개씩 빼서 비교 → 빼도 MAE·피크MAE 둘 다 안 나빠지면 '제거후보'"""
    sets = {"전체": full, **{f"-{c}": [x for x in full if x != c] for c in full}}
    E = evaluate(usable, copy_days, sets, horizon, model, seeds)
    b = E.loc["전체"]
    E["MAE변화"] = E["MAE"] - b["MAE"]
    E["피크MAE변화"] = E["피크MAE"] - b["피크MAE"]
    E["판정"] = np.where((E["MAE변화"] <= 0) & (E["피크MAE변화"] <= 0), "제거후보", "유지")
    E.loc["전체", "판정"] = "-"
    return E.sort_values("MAE변화")


def leak_test(add_candidates, cols, horizon, n=6, path=IN_XLSX, seed=0):
    """발행 시각 이후의 전력을 엉터리 값으로 바꿔도 피처 값이 그대로인지 확인.
    A: 대상일 d의 전날 23:00 이후 전력 오염 → d의 피처가 바뀌면 누수
    B: 대상 시각 T 이후 전력 오염 → T 행의 피처가 바뀌면 누수
    (생산량·기상은 합의상 사용 가능 → 검사 안 함)"""
    raw = load_data(path)
    def feats(d):
        return add_candidates(add_base_features(add_day_info(d)))
    clean = feats(raw.copy())
    rng = np.random.default_rng(seed)
    days = pd.Series(sorted(raw["date"].unique()))
    days = days[(days >= "2021-03-01") & (days <= "2021-09-10")]
    leaks = {}
    for d in rng.choice(days.values, n, replace=False):
        d = pd.Timestamp(d)
        if horizon == "A":
            cut, rows = d - pd.Timedelta(hours=1), clean["date"] == d
        else:
            hh = int(rng.integers(1, 24)); cut = d + pd.Timedelta(hours=hh)
            rows = clean["ts"] == cut
        dirty = raw.copy()
        m = dirty["ts"] >= cut
        for c in POWER_COLS:
            dirty.loc[m, c] = dirty.loc[m, c] * 3 + 777
        f2 = feats(dirty)
        for c in cols:
            a, b = clean.loc[rows, c].values, f2.loc[rows, c].values
            same = np.allclose(np.nan_to_num(a.astype(float), nan=-9e9), np.nan_to_num(b.astype(float), nan=-9e9))
            if not same:
                leaks.setdefault(c, []).append(str(d.date()))
    return leaks

FINAL_A = ["시간", "요일", "월", "생산량", "계획_전1h", "계획_후1h", "일계획합", "계획가동상태",
           "기온", "일최고기온", "습도", "시간당강수량", "가동상태래그",
           "A_plan_lead_max"]
FINAL_B = ["시간", "월", "생산량", "계획_전1h", "계획_후1h", "일계획합", "계획가동상태",
           "기온", "일최고기온", "습도", "풍속", "시간당강수량", "전력_1h전", "전력_2h전"]



# ============================================================
# 핵심 피처 조합 검증 — selection 구간만 사용
# ============================================================

PEAK_COMBO_MODEL_A = "LGB_Q65"
PEAK_COMBO_MODEL_B = "LGB_PeakW_Q65"

A_COMBOS = {
    "A_기준": FINAL_A,
    "A_+peak_rate": FINAL_A + ["A_power_weekhour4_peak_rate"],
    "A_+hour_cos": FINAL_A + ["A_hour_cos"],
    "A_+peak_rate+hour_cos": FINAL_A + [
        "A_power_weekhour4_peak_rate",
        "A_hour_cos",
    ],
}

B_COMBOS = {
    "B_기준": FINAL_B,
    "B_+hour_cos": FINAL_B + ["B_hour_cos"],
    "B_+q75": FINAL_B + ["B_power_weekhour4_q75"],
    "B_+q90": FINAL_B + ["B_power_weekhour4_q90"],
    "B_+hour_cos+q75": FINAL_B + [
        "B_hour_cos",
        "B_power_weekhour4_q75",
    ],
    "B_+hour_cos+q90": FINAL_B + [
        "B_hour_cos",
        "B_power_weekhour4_q90",
    ],
    "B_+hour_cos+q75+q90": FINAL_B + [
        "B_hour_cos",
        "B_power_weekhour4_q75",
        "B_power_weekhour4_q90",
    ],
}

def rank_peak_combos(score_table, baseline_name):
    out = score_table.copy()
    base = out.loc[baseline_name]

    out["MAE변화"] = out["MAE"] - base["MAE"]
    out["피크MAE개선"] = base["피크MAE"] - out["피크MAE"]
    out["과소P95개선"] = base["과소예측P95"] - out["과소예측P95"]
    out["Recall개선"] = out["PeakRecall"] - base["PeakRecall"]

    out["MAE_guardrail"] = out["MAE"] <= base["MAE"] + PEAK_MAE_GUARDRAIL_KW
    out["조합순위"] = np.nan

    valid = out[out["MAE_guardrail"]].copy()
    order = valid.sort_values(
        ["피크MAE", "과소예측P95", "PeakRecall"],
        ascending=[True, True, False]
    ).index

    for rank, name in enumerate(order, 1):
        out.loc[name, "조합순위"] = rank

    out["판정"] = np.where(
        out["MAE_guardrail"] &
        (out["피크MAE"] < base["피크MAE"]),
        "최종후보",
        "보류"
    )
    out.loc[baseline_name, "판정"] = "기준"

    return out.sort_values(
        ["MAE_guardrail", "조합순위"],
        ascending=[False, True]
    )

