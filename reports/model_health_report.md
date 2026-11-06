# Model health report: pd_app_v1.0

_Simulated production replay on a public dataset. Portfolio prototype, not a live lending system._

**Overall status, latest month (12): Red.** Retraining is recommended. First Red month: 10.

## What changed
- Applicant mix: the score distribution has shifted (PSI 0.44; above 0.25 means a material shift). The approval rate at the current cut-off moved from 84% in month 1 to 64%.
- The input that moved most is **EXT_SOURCE_MEAN** (CSI 0.57).
- Latest month with known outcomes (month 9): Gini 0.47 vs 0.49 at validation; predicted default rate 10.6% vs actual 10.9%.

## Alerts raised
- Month 9: **Score PSI** Amber. PSI 0.131
- Month 9: **Feature drift** Amber. EXT_SOURCE_MEAN CSI 0.167
- Month 10: **Score PSI** Red. PSI 0.255
- Month 10: **Feature drift** Red. EXT_SOURCE_3 CSI 1.579
- Month 10: **Data quality** Red. EXT_SOURCE_3 missing rate up 58pp
- Month 11: **Score PSI** Red. PSI 0.340
- Month 11: **Feature drift** Red. EXT_SOURCE_MEAN CSI 0.438
- Month 12: **Score PSI** Red. PSI 0.442
- Month 12: **Feature drift** Red. EXT_SOURCE_MEAN CSI 0.574

## Recommended actions
- Confirm with data engineering whether any input feed failed (data-quality alerts).
- If the applicant mix shift is a deliberate channel or product change, recalibrate on recent matured outcomes; if not, investigate the acquisition source.
- Keep the cut-off under review until a recalibrated model is validated.