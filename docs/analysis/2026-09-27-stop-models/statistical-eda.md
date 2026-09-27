# Actual route-label EDA: where statistical forecasts fail

Exploratory analysis on 2025-01-01–2025-10-31 organizer labels, after the existing
checksum-bound raw reconciliation (59,667,191 successful validations). No stop
pseudo-labels enter this analysis. Baseline residuals are the already disclosed
`mean_s1_d0.0` selection from the 114-configuration statistical sweep. All windows
have already been inspected; these findings are not confirmation on a blind test.

## Main finding

Unknown future operating regimes and the seasonal level/shape changes dominate
the July failure. Route 7 alone contributes 24.5% of July-window absolute error;
route 50 contributes 15.7%. A global scalar cannot fix the two: route 7 is
strongly overpredicted, while route 50 is underpredicted. The data show changes
in counts; they alone do not identify a causal service disruption or its cause.

|61-day origin|Actual count|Absolute error|Signed prediction minus actual|Score|
|---|---:|---:|---:|---:|
|May 1|11,555,908|1,391,470|174,370|0.8796|
|July 1|10,584,257|1,941,442|829,772|0.8166|
|September 1|12,754,481|1,522,283|-241,427|0.8806|

July 1 covers July 1–August 30, not August 31. May and September windows cover
complete calendar months.

## Route contributions and volume shifts

|Origin|Route|Absolute error|Share of window error|Signed bias / route actual|
|---|---:|---:|---:|---:|
|May|17|252,451|18.1%|-2.3%|
|May|12|211,895|15.2%|+0.4%|
|May|11|192,823|13.9%|+0.8%|
|May|7|186,335|13.4%|+2.4%|
|July|7|476,152|24.5%|+53.0%|
|July|50|303,977|15.7%|-10.5%|
|July|17|296,851|15.3%|+3.6%|
|July|11|225,777|11.6%|+5.8%|
|September|17|287,082|18.9%|-6.8%|
|September|50|273,985|18.0%|+10.6%|
|September|7|218,163|14.3%|+9.6%|
|September|11|193,333|12.7%|-3.3%|

June workday daily means decline against May for every active route. Workdays use
the pre-announced 2025 official civil calendar rather than a bare weekday flag.

|Route|May workday mean|June workday mean|June / May change|July workday mean|September workday mean|
|---|---:|---:|---:|---:|---:|
|1|21,699|19,659|-9.4%|17,130|21,529|
|7|28,854|25,847|-10.4%|14,575|26,277|
|11|34,534|33,034|-4.3%|29,609|34,790|
|12|38,779|35,878|-7.5%|31,865|36,753|
|17|54,717|51,652|-5.6%|46,913|53,432|
|25|7,724|6,972|-9.7%|5,766|7,626|
|26|19,213|17,148|-10.7%|14,478|19,782|
|28|11,412|10,142|-11.1%|8,306|10,298|
|50|25,859|22,721|-12.1%|25,222|24,222|

Route 5 has zero successful visible validations. Route 50's July level rises even
as most routes decline; imposing one shared seasonal multiplier is inappropriate.

Route 7 workday means: June weeks approximately 25–27k; July 14–August 10 weeks
10.5–11.0k; August 11 week 19.7k; August 25 week 21.8k. July 1–August 10 workdays
average 13,720; August 11–31 average 20,740 (+51.2%); September averages 26,277.
No days have zero counts in these intervals. August 11 is a useful descriptive
break, but these labels do not prove its cause or pre-origin knowability.

## Hourly shape and calendar effects

In July, 10:00–15:59 contributes 740,805 absolute error (38.2% of all error) and
+508,825 signed overprediction (61.3% of net overprediction). May has midday
bias +207,882 and evening 20:00–23:59 bias -145,595. September has morning
06:00–09:59 bias -201,353. These opposing signs argue for hour-specific profiles
or damped hour-band corrections, not just route daily levels.

May window bias is -123,931 in May but +298,301 in June. September window bias
is +1,331 in September and -242,758 in October. Month transitions matter even
when overall average bias appears small.

Largest May-window error days include June 14 (45,157 error, +42,539 bias) and
May 1 (31,270 error, +22,756 bias). A holiday-adjacent feature is a plausible
candidate; it is not established as a cause by this inspection. June 14 follows
the June 12–13 holiday interval and is knowable on the civil calendar.

## Proper previous-month forecasts versus NON-FORECAST oracles

The following **oracles use the target month's actual labels**. They are unavailable
at origin, must never enter model ranking or submission, and are not achievable
forecasts. They quantify in-sample residuals of these particular monthly-profile
families. They are not a mathematical upper bound on every possible model.

|Method|May origin|July origin|September origin|
|---|---:|---:|---:|
|Previous completed month route × hour mean|0.6868|0.7496|0.6769|
|Previous completed month route × weekday × hour mean, no holiday adjustment|0.7695|0.8166|0.7862|
|NON-FORECAST target-month × route × civil-daytype × hour mean|0.9083|0.8894|0.9263|
|NON-FORECAST target-month × route × civil-daytype × hour median|0.9112|0.8946|0.9281|
|NON-FORECAST target-month × route × weekday × hour mean|0.8588|0.8988|0.9347|

The proper simple monthly means are substantially weaker than a retrospectively
known future-month profile. Even a target-month civil-daytype median fails 0.93 in
all three windows. July still has roughly 10.5% absolute within-profile variation;
finer operating-regime information may matter more than another weighted mean.
The weekday oracle is worse in May because holidays with different volumes share
weekday cells; this motivates calendar-aware comparisons rather than conflating
weekday and civil day type.

## Candidates sent to the statistical agent

1. Separate June from July/August when matching historical seasons. At July origin
   June is available but already differs materially from July; any July shrinkage
   magnitude must come from earlier temporal validation, not July future labels.
2. Fit damped, clipped **per-route and per-hour-band** recent-level ratios using
   only pre-origin windows; shrink sparse ratios toward one. Explicitly compare
   with a scalar route adjustment and with no adjustment.
3. Use a past-detectable regime/recency gate. Route 7's already-observed August
   recovery may guide September, but its July collapse cannot be backfilled into
   a July-origin model from a notice or label first known after origin.
4. Add known civil-calendar holiday-adjacency and bridge-day profiles with
   shrinkage where support is small. Verify across multiple origins because
   isolated exceptional dates can make an in-sample pattern look convincing.
5. Test non-summer return profiles blended with recent route-specific levels,
   rather than treating June/July/August as one interchangeable season. Do not
   borrow actual September/October level for September forecasts.

These are exploratory, pre-origin-safe feature forms; the report does not claim
causal identification or measured improvement from implementing them. The broad
statistical experiment owns their comparative evaluation.

Reproduction used `tramflow_ml.competition.load_labels` and `complete_labels(...,
missing_as_zero=True)` with the verified organizer archive; daily workday means use
`tramflow_ml.route_models.features.OFF/WORK`. Residual source:
`/tmp/tram-statistical-sweep-final/selected_baseline_oof.csv.gz`, filtered to
`model == mean_s1_d0.0` and the three origins above. Oracles group each 61-day
window's actual rows by the stated keys, merge the mean/median back to those rows,
round once using `numpy.rint`, and compute `1 - sum(abs(y-p))/sum(y)`.
