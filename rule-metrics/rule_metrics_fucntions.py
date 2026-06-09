"""Shared helpers for fraud rule performance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

ACTION_ORDER = ["Decline", "3DS", "Alert", "Alert & 3DS", "Unknown"]

RULE_MATRIX_KEY_METRICS = [
    "daily_trigger_rate",
    "precision_of_fraud",
    "chargeback_rate",
    "success_rate",
    "unsupported_3ds_pct",
    "technical_3ds_failure_pct",
    "fraud_3ds_failure_pct",
    "abandonment_3ds_pct",
    "other_3ds_failure_pct",
    "false_positive_contribution",
    "unique_trigger_pct",
    "overlap_with_decline_pct",
    "overlap_with_3ds_pct",
    "overlap_with_alert_pct",
    "overlap_with_ml_pct",
]

RULE_MATRIX_COUNT_COLUMNS = [
    "total_triggers",
    "unique_payments",
    "sole_triggers",
    "total_fraud",
    "total_cb",
    "successful_payments",
    "total_unsupported_3ds",
    "total_technical_failures",
    "total_fraud_failures",
    "total_abandonments",
    "total_other_failures",
    "total_amount",
]

RULE_MATRIX_DATE_COLUMNS = [
    "min_date",
    "max_date",
    "days_of_evaluation",
    "rule_last_update_at",
]

AUDIT_CRITERIA = [
    {
        "level": "CRITICAL",
        "applies_to": "3DS, Alert & 3DS",
        "metric": "chargeback_rate",
        "threshold": "> 0.5%",
        "description": "Chargebacks occurring after 3DS challenge (liability leak).",
    },
    {
        "level": "CRITICAL",
        "applies_to": "3DS, Alert & 3DS",
        "metric": "success_rate",
        "threshold": "> 95%",
        "description": "Very high pass-through rate; rule may be too permissive.",
    },
    {
        "level": "CRITICAL",
        "applies_to": "3DS, Alert & 3DS",
        "metric": "unsupported_3ds_pct",
        "threshold": "> 30%",
        "description": "Large share of payments where 3DS is unsupported.",
    },
    {
        "level": "CRITICAL",
        "applies_to": "Alert, Alert & 3DS",
        "metric": "precision_of_fraud",
        "threshold": "< 2%",
        "description": "Alert queue precision below acceptable baseline.",
    },
    {
        "level": "WARNING",
        "applies_to": "Decline",
        "metric": "precision_of_fraud",
        "threshold": "< 5%",
        "description": "Decline rule catching too few confirmed fraud cases.",
    },
    {
        "level": "WARNING",
        "applies_to": "3DS, Alert & 3DS",
        "metric": "precision_of_fraud",
        "threshold": "< 1%",
        "description": "Low fraud precision on 3DS challenge cohort.",
    },
    {
        "level": "WARNING",
        "applies_to": "All actions",
        "metric": "daily_trigger_rate",
        "threshold": "< 5 triggers/day",
        "description": "Rule fires too infrequently to assess reliably.",
    },
    {
        "level": "WARNING",
        "applies_to": "All actions",
        "metric": "unique_trigger_pct",
        "threshold": "< 5%",
        "description": "Rule rarely fires alone; heavily overlaps with other rules.",
    },
    {
        "level": "WARNING",
        "applies_to": "All actions",
        "metric": "unique_trigger_pct",
        "threshold": "> 90%",
        "description": "Rule almost always fires in isolation; limited cross-rule coverage.",
    },
    {
        "level": "WARNING",
        "applies_to": "All actions",
        "metric": "overlap_with_ml_pct",
        "threshold": "> 50%",
        "description": "High co-trigger rate with ML model rules (possible redundancy).",
    },
    {
        "level": "WARNING",
        "applies_to": "All actions",
        "metric": "rule_last_update_at",
        "threshold": "> 180 days since last change",
        "description": "Rule logic has not been updated or reviewed in over 6 months.",
    },
]

ISSUE_DETAILS_WIDTH = "280px"


def rule_matrix_output_columns(include_ids: bool = True) -> list[str]:
    columns: list[str] = []
    if include_ids:
        columns += ["rule_name", "rule_action"]
    columns += RULE_MATRIX_KEY_METRICS + RULE_MATRIX_COUNT_COLUMNS + RULE_MATRIX_DATE_COLUMNS
    return columns


def categorize_detailed_failures(reason) -> str:
    r = str(reason).lower()
    if r in ["nan", "none", "", "null"]:
        return "0. Success"
    if any(
        x in r
        for x in [
            "left redirect",
            "user left",
            "preparation expired",
            "cancelled",
            "canceled",
            "session has expired",
            "challengeexpired",
            "user quit",
            "cvv modal cancelled",
        ]
    ):
        return "1. Passive Abandonment"
    if any(
        x in r
        for x in [
            "authentication has failed",
            "3d not authenticated",
            "failed 3d-secure",
            "after 3ds attempt",
            "rreq not received",
            "attempts exceeded",
        ]
    ):
        return "2. 3DS Auth Failed (Security Block)"
    if any(
        x in r
        for x in [
            "suspected fraud",
            "fraud policy",
            "risk decline",
            "risk blocked",
            "fraud suspected",
            "decline list",
            "stolen",
            "lost card",
            "security violation",
        ]
    ):
        return "3. High-Signal Fraud"
    if any(
        x in r
        for x in [
            "balance",
            "insufficient",
            "limit exceeded",
            "withdrawal amount",
            "withdrawal value",
            "amount limit",
        ]
    ):
        return "4. Financial/Limit Decline"
    if any(
        x in r
        for x in [
            "generic error",
            "y01",
            "y02",
            "invalid card",
            "technical error",
            "malfunction",
            "422",
            "not supported",
            "no security model",
            "lifecycle",
            "circuit breaker",
            "api response",
        ]
    ):
        return "5. Technical/Config Error"
    if any(
        x in r
        for x in [
            "do not honor",
            "refused",
            "declined",
            "inactive",
            "restricted",
            "blocked",
            "pick up",
        ]
    ):
        return "6. Hard Bank Decline"
    return "7. Other"


def prepare_post_update_df(df: pd.DataFrame) -> pd.DataFrame:
    df_working = df.copy()
    df_working["evaluation_date"] = pd.to_datetime(df_working["evaluation_date"]).dt.tz_localize(None)

    if "rule_last_update_at" in df_working.columns:
        df_working["rule_last_update_at"] = pd.to_datetime(
            df_working["rule_last_update_at"]
        ).dt.tz_localize(None)
        valid_rows_mask = (df_working["evaluation_date"] >= df_working["rule_last_update_at"]) | (
            df_working["rule_last_update_at"].isna()
        )
        return df_working[valid_rows_mask].reset_index(drop=True)

    return df_working.reset_index(drop=True)


def generate_rounded_rule_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = prepare_post_update_df(df)
    global_unique_payments = df_copy["payment_id"].nunique()
    if global_unique_payments == 0:
        global_unique_payments = 1

    df_copy["failure_category"] = df_copy["failure_reason"].apply(categorize_detailed_failures)

    df_copy["is_technical_failure"] = np.where(
        df_copy["failure_category"] == "5. Technical/Config Error", 1, 0
    )
    df_copy["is_fraud_failure"] = np.where(
        df_copy["failure_category"].isin(
            ["2. 3DS Auth Failed (Security Block)", "3. High-Signal Fraud"]
        ),
        1,
        0,
    )
    df_copy["is_abandonment"] = np.where(
        df_copy["failure_category"] == "1. Passive Abandonment", 1, 0
    )
    df_copy["is_other_failure"] = np.where(
        df_copy["failure_category"] == "7. Other", 1, 0
    )

    metrics_per_rule = (
        df_copy.groupby(["rule_name", "rule_action"])
        .agg(
            total_triggers=("payment_id", "count"),
            unique_payments=("payment_id", "nunique"),
            sole_triggers=("is_sole_trigger", "sum"),
            total_fraud=("is_fraud_total", lambda x: df_copy.loc[x.index].drop_duplicates("payment_id")["is_fraud_total"].sum()),
            total_cb=("is_chargeback", lambda x: df_copy.loc[x.index].drop_duplicates("payment_id")["is_chargeback"].sum()),
            successful_payments=("is_success", lambda x: df_copy.loc[x.index].drop_duplicates("payment_id")["is_success"].sum()),
            total_amount=("amount", lambda x: df_copy.loc[x.index].drop_duplicates("payment_id")["amount"].sum()),
            total_unsupported_3ds=("is_unsupported_3ds", lambda x: df_copy.loc[x.index].drop_duplicates("payment_id")["is_unsupported_3ds"].sum()),
            total_technical_failures=("is_technical_failure", lambda x: df_copy.loc[x.index].drop_duplicates("payment_id")["is_technical_failure"].sum()),
            total_fraud_failures=("is_fraud_failure", lambda x: df_copy.loc[x.index].drop_duplicates("payment_id")["is_fraud_failure"].sum()),
            total_abandonments=("is_abandonment", lambda x: df_copy.loc[x.index].drop_duplicates("payment_id")["is_abandonment"].sum()),
            total_other_failures=("is_other_failure", lambda x: df_copy.loc[x.index].drop_duplicates("payment_id")["is_other_failure"].sum()),
            min_date=("evaluation_date", "min"),
            max_date=("evaluation_date", "max"),
            overlap_decline_count=("has_decline_overlap", "sum"),
            overlap_3ds_count=("has_3ds_overlap", "sum"),
            overlap_alert_count=("has_alert_overlap", "sum"),
            
            # AGGREGATE NEW ML OVERLAP LOGIC HERE:
            overlap_ml_count=("has_ml_overlap", "sum"),
            # PRESERVE LAST UPDATE FOR TIME AUDIT ENGINE:
            rule_last_update_at=("rule_last_update_at", "max") 
        )
        .reset_index()
    )

    metrics_per_rule["min_date"] = pd.to_datetime(metrics_per_rule["min_date"])
    metrics_per_rule["max_date"] = pd.to_datetime(metrics_per_rule["max_date"])
    metrics_per_rule["days_of_evaluation"] = (
        metrics_per_rule["max_date"] - metrics_per_rule["min_date"]
    ).dt.days + 1

    metrics_per_rule["daily_trigger_rate"] = np.round(
        metrics_per_rule["total_triggers"] / metrics_per_rule["days_of_evaluation"], 2
    )
    metrics_per_rule["precision_of_fraud"] = np.round(
        (metrics_per_rule["total_fraud"] / metrics_per_rule["unique_payments"]) * 100, 2
    )
    metrics_per_rule["chargeback_rate"] = np.round(
        (metrics_per_rule["total_cb"] / metrics_per_rule["unique_payments"]) * 100, 2
    )
    metrics_per_rule["success_rate"] = np.round(
        (metrics_per_rule["successful_payments"] / metrics_per_rule["unique_payments"]) * 100,
        2,
    )
    metrics_per_rule["unique_trigger_pct"] = np.round(
        (metrics_per_rule["sole_triggers"] / metrics_per_rule["unique_payments"]) * 100, 2
    )
    metrics_per_rule["unsupported_3ds_pct"] = np.round(
        (metrics_per_rule["total_unsupported_3ds"] / metrics_per_rule["unique_payments"]) * 100,
        2,
    )
    metrics_per_rule["technical_3ds_failure_pct"] = np.round(
        (metrics_per_rule["total_technical_failures"] / metrics_per_rule["unique_payments"])
        * 100,
        2,
    )
    metrics_per_rule["fraud_3ds_failure_pct"] = np.round(
        (metrics_per_rule["total_fraud_failures"] / metrics_per_rule["unique_payments"]) * 100,
        2,
    )
    metrics_per_rule["abandonment_3ds_pct"] = np.round(
        (metrics_per_rule["total_abandonments"] / metrics_per_rule["unique_payments"]) * 100, 2
    )
    metrics_per_rule["other_3ds_failure_pct"] = np.round(
        (metrics_per_rule["total_other_failures"] / metrics_per_rule["unique_payments"]) * 100,
        2,
    )
    metrics_per_rule["overlap_with_decline_pct"] = np.round(
        (metrics_per_rule["overlap_decline_count"] / metrics_per_rule["total_triggers"]) * 100,
        2,
    )
    metrics_per_rule["overlap_with_3ds_pct"] = np.round(
        (metrics_per_rule["overlap_3ds_count"] / metrics_per_rule["total_triggers"]) * 100, 2
    )
    metrics_per_rule["overlap_with_alert_pct"] = np.round(
        (metrics_per_rule["overlap_alert_count"] / metrics_per_rule["total_triggers"]) * 100,
        2,
    )
    metrics_per_rule["false_positive_contribution"] = np.round(
        (metrics_per_rule["successful_payments"] / global_unique_payments) * 100, 2
    )

    int_cols = [
        "total_amount",
        "sole_triggers",
        "total_fraud",
        "total_cb",
        "successful_payments",
        "total_unsupported_3ds",
        "total_technical_failures",
        "total_fraud_failures",
        "total_abandonments",
        "total_other_failures",
        "days_of_evaluation",
    ]
    for col in int_cols:
        metrics_per_rule[col] = np.round(metrics_per_rule[col]).astype(int)

    metrics_per_rule["overlap_with_ml_pct"] = np.round(
        (metrics_per_rule["overlap_ml_count"] / metrics_per_rule["total_triggers"]) * 100, 2
    )

    return metrics_per_rule[rule_matrix_output_columns()].sort_values(
        by="total_triggers", ascending=False
    )


def get_audit_criteria_table() -> pd.DataFrame:
    column_order = [
        "level",
        "applies_to",
        "metric",
        "threshold",
        "description",
    ]
    level_rank = {"CRITICAL": 0, "WARNING": 1}
    return (
        pd.DataFrame(AUDIT_CRITERIA)[column_order]
        .assign(_level_rank=lambda df: df["level"].map(level_rank))
        .sort_values(["_level_rank", "applies_to", "metric"])
        .drop(columns="_level_rank")
        .reset_index(drop=True)
    )


def display_audit_criteria_reference(display, Markdown, heading_level: int = 2):
    prefix = "#" * heading_level
    display(Markdown(f"{prefix} 📏 Audit Criteria Reference"))
    display(
        Markdown(
            "Rules are flagged when they breach any threshold below. "
            "**CRITICAL** issues need immediate review; **WARNING** issues signal tuning or monitoring needs."
        )
    )

    criteria_df = get_audit_criteria_table()

    def _level_style(row: pd.Series) -> list[str]:
        if row["level"] == "CRITICAL":
            return ["background-color: #fde2e2"] * len(row)
        return ["background-color: #fff6db"] * len(row)

    styled = (
        criteria_df.style.apply(_level_style, axis=1)
        .set_properties(**{"white-space": "normal", "text-align": "left"})
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "left")]},
                {
                    "selector": "td",
                    "props": [("vertical-align", "top"), ("padding", "8px")],
                },
            ]
        )
        .hide(axis="index")
    )
    display(styled)


def run_rule_risk_audit(matrix_df: pd.DataFrame) -> pd.DataFrame:
    audit_records = []
    for _, row in matrix_df.iterrows():
        rule = row["rule_name"]
        action = row["rule_action"]
        precision = row["precision_of_fraud"]
        cb_rate = row["chargeback_rate"]
        success = row["success_rate"]
        unsupported = row["unsupported_3ds_pct"]
        unique_pct = row["unique_trigger_pct"]
        daily_rate = row["daily_trigger_rate"]
        # EXTRACT NEW METRIC VALUES
        ml_overlap_pct = row["overlap_with_ml_pct"]
        last_update = pd.to_datetime(row["rule_last_update_at"]).tz_localize(None)
        reference_now = matrix_df["max_date"].max()

        if action == "Decline":
            if precision < 5.0:
                audit_records.append(
                    {
                        "rule_name": rule,
                        "level": "WARNING",
                        "issue": f"Poor Precision ({precision}%).",
                    }
                )
        elif action in ["3DS","Alert & 3DS"]:
            if cb_rate > 0.5:
                audit_records.append(
                    {
                        "rule_name": rule,
                        "level": "CRITICAL",
                        "issue": f"Liability Leak. 3DS bypass with a {cb_rate}% chargeback rate.",
                    }
                )
            if success > 95:
                audit_records.append(
                    {
                        "rule_name": rule,
                        "level": "CRITICAL",
                        "issue": f"High success rate: {success}%",
                    }
                )
            if unsupported > 30.0:
                audit_records.append(
                    {
                        "rule_name": rule,
                        "level": "CRITICAL",
                        "issue": f"High Unsupported 3ds rate: {unsupported}%.",
                    }
                )
            if precision < 1.0:
                audit_records.append(
                    {
                        "rule_name": rule,
                        "level": "WARNING",
                        "issue": f"Low Precision:({precision}%).",
                    }
                )
        elif action in ["Alert & 3DS", "Alert"]:
            if precision < 2.0:
                audit_records.append(
                    {
                        "rule_name": rule,
                        "level": "CRITICAL",
                        "issue": f"Low precision: ({precision}%).",
                    }
                )

        if daily_rate < 5.0:
            audit_records.append(
                {
                    "rule_name": rule,
                    "level": "WARNING",
                    "issue": f"Low trigger velocity: averages only {daily_rate} triggers/day.",
                }
            )
        if unique_pct < 5.0:
            audit_records.append(
                {
                    "rule_name": rule,
                    "level": "WARNING",
                    "issue": f"High overlap: only {unique_pct}% unique footprint.",
                }
            )
        if unique_pct > 90.0:
            audit_records.append(
                {
                    "rule_name": rule,
                    "level": "WARNING",
                    "issue": f"Isolated logic: {unique_pct}% unique footprint.",
                }
            )
        # CRITERIA 1: High ML Model Overlap Redundancy Warning (>50%)
        if ml_overlap_pct > 50.0:
            audit_records.append({
                "rule_name": rule,
                "level": "WARNING",
                "issue": f"High ML Overlap: {ml_overlap_pct}%."
            })

        # CRITERIA 2: Stale Legacy Rule Configuration Logic (>5 months/150 days)
        if pd.notna(last_update):
            days_since_update = (reference_now - last_update).days
            if days_since_update > 180:
                audit_records.append({
                    "rule_name": rule,
                    "level": "WARNING",
                    "issue": f"Stale Rule: Logic has not been modified or reviewed in {days_since_update} days (>5 months)."
                })

    if not audit_records:
        return pd.DataFrame(columns=["rule_name", "level", "issue"])
    return pd.DataFrame(audit_records)


def style_comprehensive_action_table(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    top_n: int = 50,
    issue_details_width: str = ISSUE_DETAILS_WIDTH,
):
    if not audit_df.empty:
        collapsed_issues = audit_df.groupby("rule_name").agg(
            Status=(
                "level",
                lambda x: "🔴 CRITICAL" if "CRITICAL" in x.values else "🟡 WARNING",
            ),
            Issue_Details=("issue", lambda x: " | ".join(x)),
        ).reset_index()
        enriched_df = df.merge(collapsed_issues, on="rule_name", how="left")
        enriched_df["Status"] = enriched_df["Status"].fillna("🟢 HEALTHY")
        enriched_df["Issue_Details"] = enriched_df["Issue_Details"].fillna(
            "Operating within limits."
        )
    else:
        enriched_df = df.copy()
        enriched_df["Status"] = "🟢 HEALTHY"
        enriched_df["Issue_Details"] = "Operating within limits."

    for col in ("min_date", "max_date"):
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_datetime(enriched_df[col], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )

    front_cols = ["rule_name", "Status", "Issue_Details"]
    ordered_cols = [
        c
        for c in (
            RULE_MATRIX_KEY_METRICS
            + RULE_MATRIX_COUNT_COLUMNS
            + RULE_MATRIX_DATE_COLUMNS
        )
        if c in enriched_df.columns
    ]
    extra_cols = [
        c
        for c in enriched_df.columns
        if c not in front_cols + ordered_cols and c != "rule_action"
    ]
    view = (
        enriched_df[front_cols + ordered_cols + extra_cols]
        .sort_values("total_triggers", ascending=False)
        .head(top_n)
        .set_index("rule_name")
    )

    fmt = {
        "days_of_evaluation": "{:,}",
        "daily_trigger_rate": "{:,.2f}",
        "precision_of_fraud": "{:.2f}%",
        "chargeback_rate": "{:.2f}%",
        "success_rate": "{:.2f}%",
        "unsupported_3ds_pct": "{:.2f}%",
        "technical_3ds_failure_pct": "{:.2f}%",
        "fraud_3ds_failure_pct": "{:.2f}%",
        "abandonment_3ds_pct": "{:.2f}%",
        "other_3ds_failure_pct": "{:.2f}%",
        "false_positive_contribution": "{:.2f}%",
        "unique_trigger_pct": "{:.2f}%",
        "overlap_with_decline_pct": "{:.2f}%",
        "overlap_with_3ds_pct": "{:.2f}%",
        "overlap_with_alert_pct": "{:.2f}%",
        "overlap_with_ml_pct": "{:.2f}%",
        "total_amount": "{:,.0f}",
    }

    styled = view.style.format(fmt, na_rep="—")

    green_good = ["precision_of_fraud", "chargeback_rate", "fraud_3ds_failure_pct"]
    red_bad = [
        "success_rate",
        "false_positive_contribution",
        "unsupported_3ds_pct",
        "technical_3ds_failure_pct",
        "abandonment_3ds_pct",
        "other_3ds_failure_pct",
    ]
    overlap_warm = [
        "overlap_with_decline_pct",
        "overlap_with_3ds_pct",
        "overlap_with_alert_pct",
        "overlap_with_ml_pct",
    ]
    volume_blue = [
        "total_triggers",
        "unique_payments",
        "sole_triggers",
        "total_unsupported_3ds",
        "total_technical_failures",
        "total_fraud_failures",
        "total_abandonments",
        "total_other_failures",
    ]

    styled = styled.background_gradient(
        subset=[c for c in green_good if c in view.columns], cmap="Greens", axis=0
    )
    styled = styled.background_gradient(
        subset=[c for c in red_bad if c in view.columns], cmap="Reds", axis=0
    )
    styled = styled.background_gradient(
        subset=[c for c in overlap_warm if c in view.columns], cmap="YlOrRd", axis=0
    )
    styled = styled.background_gradient(
        subset=[c for c in volume_blue if c in view.columns], cmap="Blues", axis=0
    )
    styled = styled.background_gradient(
        subset=[c for c in ["unique_trigger_pct"] if c in view.columns], cmap="Purples", axis=0
    )

    column_widths = {
        "Status": {"width": "120px", "min-width": "120px", "white-space": "nowrap"},
        "Issue_Details": {
            "width": issue_details_width,
            "min-width": issue_details_width,
            "max-width": issue_details_width,
            "white-space": "normal",
            "word-wrap": "break-word",
            "overflow-wrap": "anywhere",
            "vertical-align": "top",
        },
        "min_date": {"width": "112px", "min-width": "112px"},
        "max_date": {"width": "112px", "min-width": "112px"},
    }
    for col, props in column_widths.items():
        if col in view.columns:
            styled = styled.set_properties(subset=[col], **props)

    if "Issue_Details" in view.columns:
        issue_col_idx = view.columns.get_loc("Issue_Details")
        styled = styled.set_table_styles(
            [
                {
                    "selector": f"th.col_heading.level0.col{issue_col_idx}",
                    "props": [
                        ("min-width", issue_details_width),
                        ("width", issue_details_width),
                    ],
                }
            ],
            overwrite=False,
        )

    return styled


def display_audit_exception_desk(audit_log_df: pd.DataFrame, display, Markdown, heading_level: int = 1):
    prefix = "#" * heading_level
    display(Markdown(f"{prefix} 📋 System Audit Exception Desk"))
    if audit_log_df.empty:
        display(
            Markdown(
                "🟢 **Perfect System Score:** All rules running cleanly within risk parameters."
            )
        )
        return
    display(audit_log_df.sort_values(by="level").reset_index(drop=True))


def display_action_cohort_summary(
    action: str,
    rule_performance_matrix: pd.DataFrame,
    audit_log_df: pd.DataFrame,
    display,
    Markdown,
    top_n: int = 50,
    issue_details_width: str = ISSUE_DETAILS_WIDTH,
):
    table = rule_performance_matrix[rule_performance_matrix["rule_action"] == action]
    if table.empty:
        display(Markdown(f"## 📊 {action} Cohort Summary — no rules found for this action."))
        return

    table = table.sort_values("total_triggers", ascending=False).reset_index(drop=True)
    local_rules = table["rule_name"].unique()
    local_exceptions = audit_log_df[audit_log_df["rule_name"].isin(local_rules)]
    crit_count = len(local_exceptions[local_exceptions["level"] == "CRITICAL"])
    warn_count = len(local_exceptions[local_exceptions["level"] == "WARNING"])

    display(Markdown("---"))
    display(Markdown(f"## 📊 {action} Cohort Summary — {len(table)} Registered Rules"))
    if crit_count > 0 or warn_count > 0:
        display(
            Markdown(
                f"🚨 **Advisory Note:** Found `{crit_count} Critical Exceptions` "
                f"and `{warn_count} Warnings` requiring adjustments."
            )
        )
    else:
        display(
            Markdown(
                "✅ **Cohort Healthy:** Performance profiles comply with baseline metrics."
            )
        )
    display(
        style_comprehensive_action_table(
            table,
            audit_log_df,
            top_n=top_n,
            issue_details_width=issue_details_width,
        )
    )


def audit_rule_by_country(
    source_df: pd.DataFrame,
    rule_name: str,
    display,
    Markdown,
    target_country_col: str = "buyer_country_code",
):
    df_rule = source_df[source_df["rule_name"] == rule_name].copy()
    if df_rule.empty:
        display(Markdown(f"### ❌ Audit Failed: Rule name `{rule_name}` not found in dataset."))
        return

    df_rule["evaluation_date"] = pd.to_datetime(df_rule["evaluation_date"]).dt.tz_localize(None)
    if "rule_last_update_at" in df_rule.columns and not df_rule["rule_last_update_at"].isna().all():
        df_rule["rule_last_update_at"] = pd.to_datetime(
            df_rule["rule_last_update_at"]
        ).dt.tz_localize(None)
        df_rule = df_rule[df_rule["evaluation_date"] >= df_rule["rule_last_update_at"]]

    if df_rule.empty:
        display(
            Markdown(
                f"### ⚠️ Audit Empty: No logs found for `{rule_name}` after its latest update timestamp."
            )
        )
        return

    min_d = df_rule["evaluation_date"].min()
    max_d = df_rule["evaluation_date"].max()
    days_active = (max_d - min_d).days + 1
    if days_active <= 0:
        days_active = 1

    df_rule["failure_category"] = df_rule["failure_reason"].apply(categorize_detailed_failures)
    df_rule["is_technical_failure"] = np.where(
        df_rule["failure_category"] == "5. Technical/Config Error", 1, 0
    )
    df_rule["is_fraud_failure"] = np.where(
        df_rule["failure_category"].isin(
            ["2. 3DS Auth Failed (Security Block)", "3. High-Signal Fraud"]
        ),
        1,
        0,
    )
    df_rule["is_abandonment"] = np.where(
        df_rule["failure_category"] == "1. Passive Abandonment", 1, 0
    )
    df_rule["is_other_failure"] = np.where(
        df_rule["failure_category"] == "7. Other", 1, 0
    )
    df_rule = df_rule.reset_index(drop=True)

    country_matrix = (
        df_rule.groupby(target_country_col)
        .agg(
            total_triggers=("payment_id", "count"),
            unique_payments=("payment_id", "nunique"),
            sole_triggers=("is_sole_trigger", "sum"),
            total_fraud=(
                "is_fraud_total",
                lambda x: df_rule.loc[x.index]
                .drop_duplicates("payment_id")["is_fraud_total"]
                .sum(),
            ),
            total_cb=(
                "is_chargeback",
                lambda x: df_rule.loc[x.index]
                .drop_duplicates("payment_id")["is_chargeback"]
                .sum(),
            ),
            successful_payments=(
                "is_success",
                lambda x: df_rule.loc[x.index]
                .drop_duplicates("payment_id")["is_success"]
                .sum(),
            ),
            total_amount=(
                "amount",
                lambda x: df_rule.loc[x.index].drop_duplicates("payment_id")["amount"].sum(),
            ),
            total_unsupported_3ds=(
                "is_unsupported_3ds",
                lambda x: df_rule.loc[x.index]
                .drop_duplicates("payment_id")["is_unsupported_3ds"]
                .sum(),
            ),
            total_technical_failures=(
                "is_technical_failure",
                lambda x: df_rule.loc[x.index]
                .drop_duplicates("payment_id")["is_technical_failure"]
                .sum(),
            ),
            total_fraud_failures=(
                "is_fraud_failure",
                lambda x: df_rule.loc[x.index]
                .drop_duplicates("payment_id")["is_fraud_failure"]
                .sum(),
            ),
            total_abandonments=(
                "is_abandonment",
                lambda x: df_rule.loc[x.index]
                .drop_duplicates("payment_id")["is_abandonment"]
                .sum(),
            ),
            total_other_failures=(
                "is_other_failure",
                lambda x: df_rule.loc[x.index]
                .drop_duplicates("payment_id")["is_other_failure"]
                .sum(),
            ),
        )
        .reset_index()
    )

    country_matrix["daily_trigger_rate"] = np.round(
        country_matrix["total_triggers"] / days_active, 2
    )
    country_matrix["precision_of_fraud"] = np.round(
        (country_matrix["total_fraud"] / country_matrix["unique_payments"]) * 100, 2
    )
    country_matrix["chargeback_rate"] = np.round(
        (country_matrix["total_cb"] / country_matrix["unique_payments"]) * 100, 2
    )
    country_matrix["success_rate"] = np.round(
        (country_matrix["successful_payments"] / country_matrix["unique_payments"]) * 100, 2
    )
    country_matrix["unique_trigger_pct"] = np.round(
        (country_matrix["sole_triggers"] / country_matrix["unique_payments"]) * 100, 2
    )
    country_matrix["unsupported_3ds_pct"] = np.round(
        (country_matrix["total_unsupported_3ds"] / country_matrix["unique_payments"]) * 100, 2
    )
    country_matrix["technical_3ds_failure_pct"] = np.round(
        (country_matrix["total_technical_failures"] / country_matrix["unique_payments"]) * 100,
        2,
    )
    country_matrix["fraud_3ds_failure_pct"] = np.round(
        (country_matrix["total_fraud_failures"] / country_matrix["unique_payments"]) * 100, 2
    )
    country_matrix["abandonment_3ds_pct"] = np.round(
        (country_matrix["total_abandonments"] / country_matrix["unique_payments"]) * 100, 2
    )
    country_matrix["other_3ds_failure_pct"] = np.round(
        (country_matrix["total_other_failures"] / country_matrix["unique_payments"]) * 100, 2
    )

    country_matrix = country_matrix.sort_values(by="total_triggers", ascending=False).reset_index(
        drop=True
    )
    country_matrix.rename(columns={target_country_col: "Country"}, inplace=True)

    int_cols = [
        "total_amount",
        "sole_triggers",
        "total_fraud",
        "total_cb",
        "successful_payments",
        "total_unsupported_3ds",
        "total_technical_failures",
        "total_fraud_failures",
        "total_abandonments",
        "total_other_failures",
    ]
    for col in int_cols:
        country_matrix[col] = country_matrix[col].astype(int)

    col_order = [
        "Country",
        "total_triggers",
        "unique_payments",
        "daily_trigger_rate",
        "precision_of_fraud",
        "chargeback_rate",
        "success_rate",
        "unique_trigger_pct",
        "unsupported_3ds_pct",
        "technical_3ds_failure_pct",
        "fraud_3ds_failure_pct",
        "abandonment_3ds_pct",
        "other_3ds_failure_pct",
        "total_amount",
    ]
    country_matrix = country_matrix[col_order]

    min_d_str = pd.Timestamp(min_d).strftime("%Y-%m-%d")
    max_d_str = pd.Timestamp(max_d).strftime("%Y-%m-%d")
    display(Markdown(f"## 🗺️ Country Deep-Dive Profile: `{rule_name}`"))
    display(
        Markdown(
            f"Evaluating operational window of **{days_active} day(s)** "
            f"from `{min_d_str}` to `{max_d_str}`."
        )
    )

    fmt = {
        "total_triggers": "{:,}",
        "unique_payments": "{:,}",
        "daily_trigger_rate": "{:,.2f}",
        "precision_of_fraud": "{:.2f}%",
        "chargeback_rate": "{:.2f}%",
        "success_rate": "{:.2f}%",
        "unique_trigger_pct": "{:.2f}%",
        "unsupported_3ds_pct": "{:.2f}%",
        "technical_3ds_failure_pct": "{:.2f}%",
        "fraud_3ds_failure_pct": "{:.2f}%",
        "abandonment_3ds_pct": "{:.2f}%",
        "other_3ds_failure_pct": "{:.2f}%",
        "total_amount": "${:,.0f}",
    }

    styled_table = (
        country_matrix.set_index("Country")
        .style.format(fmt)
        .background_gradient(
            subset=["precision_of_fraud", "fraud_3ds_failure_pct"], cmap="Greens"
        )
        .background_gradient(
            subset=[
                "success_rate",
                "unsupported_3ds_pct",
                "technical_3ds_failure_pct",
                "abandonment_3ds_pct",
            ],
            cmap="Reds",
        )
        .background_gradient(subset=["total_triggers", "unique_payments"], cmap="Blues")
        .background_gradient(subset=["unique_trigger_pct"], cmap="Purples_r")
    )
    display(styled_table)


def analyze_single_rule_overlap(
    df: pd.DataFrame,
    target_rule: str,
    display,
    Markdown,
    min_shared_triggers: int = 1,
):
    df_clean = prepare_post_update_df(df)
    target_payments = df_clean[df_clean["rule_name"] == target_rule]["payment_id"].unique()
    target_total_volume = len(target_payments)

    if target_total_volume == 0:
        display(
            Markdown(
                f"### ❌ Execution Stopped: Rule `{target_rule}` has zero registered triggers "
                "in the post-update evaluation window."
            )
        )
        return

    co_occurring_records = df_clean[
        (df_clean["payment_id"].isin(target_payments)) & (df_clean["rule_name"] != target_rule)
    ]
    if co_occurring_records.empty:
        display(
            Markdown(
                f"### ✅ Perfect Isolation: Rule `{target_rule}` has **0% overlap** "
                "with any other rule in the system!"
            )
        )
        return

    overlap_summary = (
        co_occurring_records.groupby(["rule_name", "rule_action"])
        .agg(shared_count=("payment_id", "nunique"))
        .reset_index()
    )
    overlap_summary = overlap_summary[overlap_summary["shared_count"] >= min_shared_triggers]
    if overlap_summary.empty:
        display(
            Markdown(
                f"### ✅ Low Exposure: No other rules share more than {min_shared_triggers} "
                f"triggers with `{target_rule}`."
            )
        )
        return

    overlap_summary["overlap_pct"] = np.round(
        (overlap_summary["shared_count"] / target_total_volume) * 100, 2
    )
    overlap_summary = overlap_summary.sort_values(by="overlap_pct", ascending=False)

    display(Markdown(f"## 🔍 Targeted Overlap Analysis: `{target_rule}`"))
    display(
        Markdown(
            f"Rule `{target_rule}` caught **{target_total_volume:,} unique payments** in total "
            "during its post-update window. The chart below shows which other rules fired on "
            "those same transactions:"
        )
    )

    fig = px.bar(
        overlap_summary,
        x="overlap_pct",
        y="rule_name",
        color="rule_action",
        orientation="h",
        text="overlap_pct",
        title=f"Co-Trigger Surface Mapping for Rule: {target_rule}",
        labels={
            "rule_name": "Overlapping Rule Name",
            "overlap_pct": f"% of {target_rule} Transactions Also Caught by This Rule",
            "rule_action": "Action Type",
        },
        color_discrete_sequence=px.colors.qualitative.Safe,
        custom_data=["shared_count"],
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            texttemplate="%{x:.1f}%",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                f"<b>Overlapping Rule:</b> %{{y}}<br>"
                f"<b>Action Cohort:</b> {trace.name}<br>"
                f"<b>Overlap Strength:</b> %{{x:.2f}}%<br>"
                f"<b>Absolute Intersections:</b> %{{customdata[0]:,.0f}} shared payments<extra></extra>"
            ),
        )
    )
    dynamic_height = max(400, len(overlap_summary) * 25)
    max_overlap = overlap_summary["overlap_pct"].max()
    fig.update_layout(
        width=1000,
        height=dynamic_height,
        xaxis=dict(
            ticksuffix="%",
            range=[0, max(max_overlap * 1.18 + 3, 10)],
            showgrid=True,
            gridcolor="#e6e6e6",
        ),
        yaxis=dict(categoryorder="total descending"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        margin=dict(l=200, r=60, t=80, b=50),
    )
    fig.show()


def plot_interactive_heatmap_chunk(
    df: pd.DataFrame, start_rank: int = 0, end_rank: int = 20
):
    top_ranked_rules = df["rule_name"].value_counts().index.tolist()
    selected_subset = top_ranked_rules[start_rank:end_rank]

    if not selected_subset:
        print(f"No rules found within rank range {start_rank} to {end_rank}.")
        return

    filtered_df = df[df["rule_name"].isin(selected_subset)]
    indicator_matrix = pd.crosstab(filtered_df["payment_id"], filtered_df["rule_name"]).map(
        lambda x: 1 if x > 0 else 0
    )
    co_occurrence = indicator_matrix.T.dot(indicator_matrix)

    row_totals = co_occurrence.values.diagonal()
    row_totals_safe = [tot if tot > 0 else 1 for tot in row_totals]
    overlap_pct_matrix = (co_occurrence.T / row_totals_safe).T * 100
    overlap_pct_matrix = overlap_pct_matrix.reindex(index=selected_subset, columns=selected_subset)

    fig = px.imshow(
        overlap_pct_matrix,
        labels=dict(x="Secondary Rule triggered", y="Primary Rule triggered", color="% Overlap"),
        x=selected_subset,
        y=selected_subset,
        color_continuous_scale="YlOrRd",
        zmin=0,
        zmax=100,
        title=f"Interactive Overlap Surface Matrix — Rules Ranked {start_rank + 1} to {end_rank}",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Primary Rule:</b> %{y}<br>"
            "<b>Secondary Rule:</b> %{x}<br>"
            "<b>Overlap Frequency:</b> %{z:.1f}%<extra></extra>"
        )
    )
    fig.update_layout(width=850, height=750, xaxis_tickangle=-45, title_font=dict(size=14, family="Arial"))
    fig.show()


def _import_network_graph_deps():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "plot_cohort_network_graph requires optional packages. "
            "Install them with: pip install networkx matplotlib"
        ) from exc
    return nx, plt, Line2D


def plot_cohort_network_graph(
    df: pd.DataFrame,
    target_actions: str | list[str] | None = None,
    overlap_threshold_pct: float = 25.0,
    min_triggers: int = 15,
):
    nx, plt, Line2D = _import_network_graph_deps()

    if target_actions is None:
        cohort_df = df.copy()
        cohort_label = "All Actions"
    elif isinstance(target_actions, str):
        cohort_df = df[df["rule_action"] == target_actions]
        cohort_label = target_actions
    else:
        cohort_df = df[df["rule_action"].isin(target_actions)]
        cohort_label = ", ".join(target_actions)

    if cohort_df.empty:
        print(f"No rules found matching action type(s): {target_actions or 'all actions'}")
        return

    rule_counts = cohort_df["rule_name"].value_counts()
    active_rules = rule_counts[rule_counts >= min_triggers].index
    filtered_df = cohort_df[cohort_df["rule_name"].isin(active_rules)]
    if filtered_df.empty:
        print(
            f"No rules in the '{cohort_label}' cohort met the minimum trigger threshold of {min_triggers}."
        )
        return

    rule_action_map = (
        filtered_df.drop_duplicates("rule_name").set_index("rule_name")["rule_action"].to_dict()
    )

    indicator_matrix = pd.crosstab(filtered_df["payment_id"], filtered_df["rule_name"]).map(
        lambda x: 1 if x > 0 else 0
    )
    co_occurrence = indicator_matrix.T.dot(indicator_matrix)

    graph = nx.Graph()
    connected_nodes = set()
    edges_to_add = []

    for i in range(len(co_occurrence.columns)):
        for j in range(i + 1, len(co_occurrence.columns)):
            r1 = co_occurrence.index[i]
            r2 = co_occurrence.columns[j]
            shared_count = co_occurrence.iloc[i, j]
            if shared_count <= 0:
                continue
            max_possible_overlap = (
                max(shared_count / co_occurrence.iloc[i, i], shared_count / co_occurrence.iloc[j, j])
                * 100
            )
            if max_possible_overlap >= overlap_threshold_pct:
                edges_to_add.append((r1, r2, shared_count, max_possible_overlap))
                connected_nodes.update([r1, r2])

    if not edges_to_add:
        print(
            f"Perfect operational separation! No '{cohort_label}' rules overlap more than "
            f"{overlap_threshold_pct}%."
        )
        return

    for rule in connected_nodes:
        graph.add_node(rule, size=int(co_occurrence.loc[rule, rule]))
    for r1, r2, shared, pct in edges_to_add:
        graph.add_edge(r1, r2, weight=shared, pct=pct)

    action_palette = {
        "Decline": "#4C78A8",
        "3DS": "#F58518",
        "Alert": "#54A24B",
        "Alert & 3DS": "#E45756",
        "Unknown": "#B279A2",
    }
    node_colors = [
        action_palette.get(rule_action_map.get(node, "Unknown"), "#888888")
        for node in graph.nodes()
    ]

    plt.figure(figsize=(11, 9))
    pos = nx.spring_layout(graph, k=0.85, iterations=60, seed=42)
    raw_sizes = [nx.get_node_attributes(graph, "size")[node] for node in graph.nodes()]
    node_sizes = [max(150, min(1500, size * 2.5)) for size in raw_sizes]
    edges = graph.edges()
    weights = [graph[u][v]["pct"] / 8 for u, v in edges]

    nx.draw_networkx_nodes(
        graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9
    )
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=weights, edge_color="#B0B0B0", alpha=0.6)
    labels = nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")
    for label in labels.values():
        label.set_bbox(dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"))

    present_actions = sorted(
        {rule_action_map.get(node, "Unknown") for node in graph.nodes()},
        key=lambda action: ACTION_ORDER.index(action) if action in ACTION_ORDER else len(ACTION_ORDER),
    )
    if len(present_actions) > 1:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=action_palette.get(action, "#888888"),
                label=action,
                markersize=10,
            )
            for action in present_actions
        ]
        plt.legend(handles=legend_handles, title="Rule Action", loc="upper right")

    plt.title(
        f"'{cohort_label}' Cohort Dependency Network\n(Showing Intersections >= {overlap_threshold_pct}%)",
        fontsize=12,
        weight="bold",
        pad=15,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_3ds_failure_distribution_bars(
    df: pd.DataFrame,
    target_actions=None,
    min_triggers: int = 10,
):
    if target_actions is None:
        target_actions = ["3DS", "Alert & 3DS"]

    cohort_df = df[df["rule_action"].isin(target_actions)].copy()
    if cohort_df.empty:
        print(f"No records found matching actions: {target_actions}")
        return

    rule_counts = cohort_df["rule_name"].value_counts()
    active_rules = rule_counts[rule_counts >= min_triggers].index
    filtered_df = cohort_df[cohort_df["rule_name"].isin(active_rules)].copy()
    if filtered_df.empty:
        print(f"No rules inside {target_actions} crossed the minimum volume of {min_triggers} triggers.")
        return

    filtered_df["failure_category"] = filtered_df["failure_reason"].apply(categorize_detailed_failures)
    distribution_counts = filtered_df.groupby(["rule_name", "failure_category"]).size().reset_index(name="count")
    rule_totals = filtered_df.groupby("rule_name").size().reset_index(name="total_rule_triggers")
    merged_distribution = distribution_counts.merge(rule_totals, on="rule_name")
    merged_distribution["Percentage"] = np.round(
        (merged_distribution["count"] / merged_distribution["total_rule_triggers"]) * 100, 2
    )
    merged_distribution = merged_distribution.sort_values(by="total_rule_triggers", ascending=True)

    fig = px.bar(
        merged_distribution,
        x="Percentage",
        y="rule_name",
        color="failure_category",
        orientation="h",
        custom_data=["count"],
        labels={
            "rule_name": "Rule Configuration",
            "Percentage": "Relative Share of Triggered Volume (%)",
            "failure_category": "Mapped Root Cause",
        },
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            hovertemplate=(
                f"<b>Rule:</b> %{{y}}<br>"
                f"<b>Category:</b> {trace.name}<br>"
                f"<b>Share:</b> %{{x:.1f}}%<br>"
                f"<b>Absolute Count:</b> %{{customdata[0]:,.0f}} triggers<extra></extra>"
            )
        )
    )
    calculated_height = max(500, len(active_rules) * 28)
    fig.update_layout(
        barmode="stack",
        xaxis=dict(ticksuffix="%", range=[0, 100]),
        height=calculated_height,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=200, r=30, t=100, b=50),
    )
    fig.show()


def plot_alert_precision_scatter_log(
    matrix_df: pd.DataFrame,
    target_actions=None,
):
    if target_actions is None:
        target_actions = ["Alert", "Alert & 3DS"]

    plot_df = matrix_df[matrix_df["rule_action"].isin(target_actions)].copy()
    if plot_df.empty:
        print(f"No matrix records found matching action profiles: {target_actions}")
        return

    plot_df["display_daily_rate"] = plot_df["daily_trigger_rate"].apply(
        lambda x: 0.01 if x <= 0 else x
    )
    max_triggers = plot_df["total_triggers"].max() if not plot_df.empty else 1
    plot_df["bubble_display_size"] = plot_df["total_triggers"].apply(
        lambda x: (x / max_triggers) * 40 + 8
    )

    fig = px.scatter(
        plot_df,
        x="display_daily_rate",
        y="precision_of_fraud",
        size="bubble_display_size",
        color="rule_action",
        text="rule_name",
        title="Alert Queue Analysis: Daily Trigger Rate vs. Fraud Precision (Logarithmic Spacing)",
        labels={
            "display_daily_rate": "Daily Trigger Volume Rate (Count / Day) — Log Scale",
            "precision_of_fraud": "Precision of Fraud (%)",
            "rule_action": "Rule Setup Type",
        },
        color_discrete_map={"Alert": "#32628d", "Alert & 3DS": "#e45756"},
        hover_name="rule_name",
        log_x=True,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Rule:</b> %{hovertext}<br>"
            "<b>True Daily Rate:</b> %{customdata[2]:,.2f} triggers/day<br>"
            "<b>Fraud Precision:</b> %{y:.2f}%<br>"
            "<b>Total Lifetime Triggers:</b> %{customdata[0]:,}<br>"
            "<b>Unique Payments Hit:</b> %{customdata[1]:,}<extra></extra>"
        ),
        customdata=np.stack(
            (plot_df["total_triggers"], plot_df["unique_payments"], plot_df["daily_trigger_rate"]),
            axis=-1,
        ),
        textposition="top center",
    )
    fig.add_hline(
        y=2.0,
        line_dash="dash",
        line_color="#bf5858",
        opacity=0.7,
        annotation_text="Critical Floor (2% Precision)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        width=1100,
        height=720,
        xaxis=dict(showgrid=True, gridcolor="#e6e6e6", dtick=1, tickformat=",.2f"),
        yaxis=dict(showgrid=True, gridcolor="#e6e6e6", ticksuffix="%", range=[-5, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
    )
    fig.show()
