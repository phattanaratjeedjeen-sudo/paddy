# Q&A Notes — HW1 (Paddy yield regression)

Running log of questions asked about `src/paddy_ML_regression.ipynb` and their answers.

---

## Q: what does it mean "one row = one paddy plot's cultivation-season record (1–6 ha) in one of six agri-blocks."

**A:** This is the sample definition from §1.2. It defines the unit of observation:

- **one paddy plot** — a specific piece of farmland, not a farm or a farmer.
- **cultivation-season record** — all the data for that plot covers one growing cycle (planting through ~day 120): inputs used, weather experienced, and final yield.
- **(1–6 ha)** — plot size is always a whole number from 1 to 6 hectares; no other sizes appear in the data.
- **in one of six agri-blocks** — every plot belongs to exactly one of six named districts (Chinnasalem, Cuddalore, Kallakurichi, Kurinjipadi, Panruti, Sankarapuram).

So each of the 2,789 rows is one plot, in one of 6 districts, sized 1–6 ha, tracked for one season.

**Why it matters:** it sets the boundaries of what the model can speak to — only 1–6 ha plots, only these six districts, only one season. Extrapolating beyond that (e.g. a 10 ha plot, a different region, a drought year) is not supported by the data. This is rubric item 1 (Guideline: "What is the definition of your Samples?") and underlies the extrapolation warning in §17 (Discussion).

---

## Q: what is block-level, plot-level "Weather columns are block-level summaries, not plot-level measurements."

**A:** This is assumption A3 from §1.3 — about the granularity at which a measurement was actually taken.

- **Plot-level** would mean each individual plot has its own value, measured separately on that specific land.
- **Block-level** means one value is recorded per district (`Agriblock`) and every plot in that district shares the same number, as if from one shared weather station.

A plot-level value could explain why one specific plot differed from its neighbour; a block-level value can't — it's really a property of the district, copied onto every row in it.

**Verified directly in §10** (not just assumed): for all 24 weather columns and the 4 wind-direction columns, every plot within a given agri-block has the identical value. Only 6 distinct weather "profiles" exist for 6 agri-blocks, mapped one-to-one (Cramér's V = 1.0).

**Consequence:** weather columns can't explain within-district yield differences (same value for every plot in a district), and §8–§9 already showed no between-district relationship either. This is the mechanism behind §10's redundancy finding — 24 "independent" weather columns collapse to one relabeling of the 6-category district variable.

---

## Q: what is OLS in A2

**A:** OLS = Ordinary Least Squares — the fitting method behind "linear regression" here. It finds the coefficients **a** that minimize the sum of squared residuals, L(a) = ||Xa - y||^2 (Ch01 formula, worked through in §14.1 two ways: the closed-form normal equation and gradient descent, confirmed to agree).

In A2, "OLS inference and K-fold CV assume independent observations" refers to one of OLS's underlying assumptions (separate from the residual normality/constant-variance assumptions tested in §14.5): knowing the error on one row should tell you nothing about the error on another. If rows were secretly linked (e.g. repeated measurements of the same plot across years, or time-ordered rows), OLS's inference and K-fold CV's generalization estimate would be too optimistic.

Why it's an assumption, not a checked fact: there's no plot ID, date, or year column, so it can't be verified directly. It's partially checked in §14.5 via the Durbin-Watson statistic (~2, no evidence of row-order autocorrelation) and relied on by the train/test split in §3.

---

## Q: defination of data leakage

**A:** Data leakage = when information that would not actually be available at prediction time (or that only exists because of the outcome itself) gets into model training or evaluation. The model looks much better than it really is on test data, but performs worse in real deployment, since that information isn't there yet when you'd actually need to predict.

Core test: "Would I know this value at the moment I actually need to make the prediction?" If no, it's leakage.

Notebook's own definition (§1.4): "Target leakage = a predictor that contains information about the target that would not be available when the model is used, so it inflates test scores but fails in practice."

Four forms identified/controlled in this notebook:
1. Target/feature leakage — a column that only exists because of the outcome or is measured at/after it (e.g. `Trash`, a harvest by-product — excluded in §1.4, confirmed worthless in §8).
2. Preprocessing leakage — fitting scaler/imputer/feature-selection stats using test rows. Controlled by fitting everything on `train` only (§3, §12).
3. Duplicate leakage — an identical row in both train and test. Handled by removing duplicates before splitting (§2-§3); cost quantified (25.8% of test rows would've had an exact twin in training otherwise).
4. Group/selection leakage — a grouping structure (e.g. `Agriblock`, same weather for every plot in it) leaks info between train/test via a random split. Checked with leave-one-block-out CV (§14.4) and a leaky-vs-honest feature-selection demo (§14.6).

Common thread: leakage makes the test score lie — optimistic exactly where you built the test set to catch that.

---

## Q: identical rows falling in both train and test are removed before the split what is this

**A:** From §1.4's leakage-control list, describing what §2-§3 actually do.

"Identical rows" = exact duplicate records — every one of the 45 columns matches another row exactly. 451 such rows exist (16.2% of the file).

Why it's a problem: if duplicate row #7 lands in train and its identical twin #812 lands in test, the model effectively trains on the exact answer for a row it's later "tested" on — it's memorization, not generalization, and it inflates the test score.

Why the order matters ("removed BEFORE the split"): the notebook (1) deduplicates the whole dataset first (§2: 2,789 -> 2,338 rows), then (2) splits the already-deduplicated data 70/30 (§3). If you split first and deduplicated train/test separately instead, a duplicate pair straddling both sides would never be caught, since each side's dedup only looks at itself.

Cost of getting it wrong, quantified (§3): splitting the original un-deduplicated data with the same seed shows 25.8% of test rows would have an exact twin in training.

---

## Q: (pasted output) rows involved in a duplicate group: 812 | distinct duplicated records: 361 / group sizes {2:293,3:50,4:14,5:4} / Distinct feature vectors: 411 for 2789 rows -> avg 6.8 rows share one feature vector ...but avg 5.7 distinct yields, target NOT a function of features (irreducible noise) — give more detail

**A:** Two separate checks from §2, easy to conflate.

**Exact duplicates (all 45 columns incl. target):**
- 812 = every row that's part of some duplicate group (all copies, not just "extras").
- 361 = distinct row-patterns that repeat.
- group sizes {2:293, 3:50, 4:14, 5:4}: 293 patterns appear twice, 50 appear 3x, 14 appear 4x, 4 appear 5x.
- Check: 293*2+50*3+14*4+4*5 = 812 (self-consistent).
- Removing all but one copy of each of the 361 patterns removes 812-361=451 rows — matches the earlier "451 duplicates removed" figure.

**Predictor-only collisions (44 columns, target excluded) — a different, broader question:**
- 411 distinct feature vectors for 2,789 rows -> only 411 distinct combinations of predictors exist.
- avg 6.8 rows share one feature vector -> ~7 rows on average have identical inputs (same Hectares, Variety, fertilizer, weather, everything).
- ...but avg 5.7 distinct yields among those ~7 rows -> identical inputs still produce different yields.

**Why this matters:** if two rows share identical predictors but different yields, no function of those predictors (however good the model) can predict both correctly. This sets a hard ceiling on achievable accuracy — "irreducible error" / noise floor, a property of the data, not a modeling failure. Distinct from the exact-duplicate cleanup (which is about leakage between train/test); this check is about honestly acknowledging, before any model is fit, that R²=1.00 was never achievable — relevant later when the final model reaches R²≈0.99.

---

## Q: plots is แปลงปลูก right?

**A:** Yes. แปลง = plot/parcel of land, ปลูก = to plant/cultivate, so แปลงปลูก = "planting/cultivation plot" — exactly the "plot" used throughout the notebook (e.g. §1.2's sample definition).

This term is from the original Thai dataset README: "...ประกอบด้วยข้อมูลพื้นที่และแปลงปลูก พันธุ์ข้าว..." ("...contains info on area and cultivation plots, rice variety...") and row 2 of the original variable table: "เขต/บล็อกการเกษตรที่แปลงตั้งอยู่" ("district/agri-block where the plot is located") — which became Agriblock's English definition when translated.

So แปลงปลูก (or แปลง for short) -> "plot" is correct, and matches §1.2's unit of observation.

---

## Q: categorical if < 20 unique values and values repeat which file or page refer to this statement

**A:** guid/HW_ML_Guideline.pdf, page 1, section 4 ("Visual Exploratory Data Analysis for data distribution and feature selection based on data distribution") -> "Categorical variables" -> "Guidelines".

Exact wording: "We can spot a categorical variable in the data by looking at the unique values in them. Typically a categorical variable contains less than 20 Unique values AND there is repetition of values, which means the data can be grouped by those unique values."

Referenced in notebook §2.1 ("Data representation and scale types") — applied together with domain meaning, not mechanically, because this dataset is unusually coarse (most numeric columns also have <20 unique values purely from being templated/per-hectare-dose data, §1.3 A5), so applying the rule literally would wrongly reclassify things like Hectares or rainfall as categorical. §2.1's code checks this directly (36 of 37 "quantitative" columns have <20 unique values) and explicitly does not reclassify them.

---

## Q: meaing of scale types

**A:** "Scale type" (level of measurement) classifies what math/comparisons are valid for a variable. From Ch01 slides (~33-35), used in notebook §2.1 to decide encoding/transform choices.

Four levels, each adding more allowed operations:
- **Nominal**: only equality (=, !=) — unordered categories. E.g. Wind Direction, Variety. No arithmetic, no order.
- **Ordinal**: equality + order (<, >). E.g. education level, a 1-5 rating. Gaps between values aren't meaningful numbers.
- **Interval**: order + meaningful differences, but NO true zero. E.g. temperature in °C. Can't take ratios (20°C isn't "twice as hot" as 10°C).
- **Ratio**: everything above + a true zero, so ratios are meaningful. E.g. Hectares, kg, mm rain, age. Least restrictive.

Why it matters in the notebook (§2.1):
- Nominal columns -> must be one-hot encoded, never integer-coded (integer codes would invent a fake order/distance a linear model would wrongly learn). Applies to Variety, Agriblock, Soil Types, Nursery, 4 Wind Direction columns (§12).
- No ordinal columns exist here, so ordinal encoding is correctly never used.
- Temperature columns are interval-scaled (no true zero) -> ratio-based transforms (log, cube-root) would be meaningless on them.
- Most numeric columns (area, kg, mm, %) are ratio-scaled -> this is what makes log-transforming the target (Paddy yield) legitimate to consider in §4/§13.

So scale type is the justification behind which encoding/transform choices are valid, not just taxonomy.

---

## Q: meaning of have a true zero

**A:** "True zero" = the value 0 genuinely means "none of this quantity exists," not just an arbitrary reference point.

Examples: Hectares=0 or yield=0 -> literally none -> true zero (ratio-scaled). Temperature 0°C -> does NOT mean "no heat" (just water's freezing point, arbitrary; actual "no heat" = -273.15°C) -> no true zero (interval-scaled).

Why it matters: true zero is what makes ratios ("twice as much") meaningful. 20kg really is 2x 10kg in any unit (grams, pounds) since zero mass means nothing everywhere. But 20°C isn't 2x as hot as 10°C — convert to °F (68 vs 50) and the ratio breaks, because Celsius's zero was arbitrary.

Notebook connection: this is why log/cube-root/Box-Cox transforms (§4/§13) are restricted to ratio-scaled columns (area, kg, mm rain — real true-zero quantities) and only standardisation (no ratio claims) is used on interval-scaled temperature columns (§2.1) — log transforms are fundamentally about ratios, which are meaningless without a true zero.

---

## Q: **Consequences for the analysis (reasons for later choices).** this topic describe about data attribute and proper scale tools?

**A:** Yes, exactly. That section bridges (1) classification — what scale type each attribute is (Nominal/Ordinal/Interval/Ratio, the table above it) — with (2) methodology — what tool/technique is actually valid given that classification.

General principle: scale type constrains which statistical operations are mathematically meaningful on a variable. The section is a practical checklist turning that theory into concrete later decisions:
- Can I average/order this column? -> scale type decides.
- One-hot vs integer encoding for a category? -> Nominal (no order) = one-hot only (§12).
- Can I take a ratio/log of this column? -> needs a true zero -> Ratio yes, Interval no (§4/§13 target transform; §2.1 excludes temperature).
- Is ordinal encoding needed? -> only if an Ordinal attribute exists -> none here, correctly skipped.

So: yes, it's about matching each attribute's type to the appropriate analysis tool, rather than applying one method blindly to the whole dataset regardless of data type.

---

## Q: where Kolmogorov–Smirnov test comes from

**A:** Not from the course slides or Guideline PDF — doesn't appear in Ch01/Ch02/HW_ML_Guideline. Added independently in §3 as an extra rigor check beyond what the course materials explicitly required.

General origin: standard statistical test (Kolmogorov 1933; two-sample form by Smirnov 1939). Compares two samples' empirical CDFs, finds the largest vertical gap between them (statistic D). Small D / p>0.05 means no evidence the samples differ. Non-parametric — doesn't assume any distribution shape, so it works even though §4 shows the target isn't normal.

Where used in notebook (§3): right after the 70/30 train/test split, to check the split didn't put a biased subset of yields into either side. Result: D=0.0308, p=0.724 (no evidence of difference) — basis for §3's "target distributions are statistically indistinguishable" claim.

Why it matters: Ch01 says to split 70/30 but gives no tool to confirm the split worked as intended — KS test fills that gap, following the notebook's general principle of testing claims rather than asserting them.

---

## Q: what are the assumptions or limitations to use Kolmogorov–Smirnov test

**A:** KS test assumptions/limitations and how §3's use holds up:

1. **Continuous, no ties** — theory assumes continuous variable, no exact repeats. Partially violated here: yield has only 477 distinct values over 2,789 rows (multi-modal by plot size, §4). Ties make the test conservative (weaker, not wrong).
2. **Independent, i.i.d. samples** — same caveat as A2 (§1.3): no plot ID, can't fully verify independence.
3. **Random samples from each population** — satisfied (train_test_split, random_state=42).
4. **Most sensitive near center, weak in tails** — KS statistic = largest CDF gap, usually near median. Relevant since §6 found the worst heteroscedasticity at the largest plot sizes — exactly where KS is weakest. A KS pass doesn't guarantee tail match.
5. **Omnibus test of whole distribution** — significant = "differ somehow" (not how); non-significant = "no difference detected" (not "identical"). Notebook's wording is careful about this ("no evidence the parts differ", not "identical").
6. **Univariate only** — only checks the target's marginal distribution, says nothing about predictors jointly. This is why §3 pairs KS with a separate Hectares-composition table check.

Bottom line: KS result (D=0.031, p=0.724) is legitimate but partial evidence — weakened by ties, blind to tails, silent on predictors — which is why it's presented alongside the Hectares table, not as standalone proof.

---

## Q: why set random state = 42

**A:** SEED=42 set once in §0, reused everywhere randomness occurs (train_test_split §3, KFold §14.3, permutation tests for MI noise floor §8/§11, synthetic outlier injection §6).

Why fix a seed at all: without it, re-running the notebook gives different numbers every time (different split, different CV folds, different R²) — breaks reproducibility (stated in §0: "fixed random_state, so re-running reproduces them"). Also makes model comparisons (M1-M4, §14.2) fair — same split for all.

Why specifically 42: arbitrary — any fixed integer works equally well. It's a common ML/data-science convention (Hitchhiker's Guide to the Galaxy reference), used constantly in sklearn docs/tutorials/Kaggle. Carries no statistical meaning, doesn't favor any result.

Honesty check: seed was set once upfront, never changed/tuned to chase a better-looking result (that would be a subtle form of overfitting to the split). This is also why the notebook doesn't rely on one split alone — §14.3 (5-fold CV) and §14.4 (leave-one-block-out CV) re-check results under different splits to confirm they aren't a seed artifact.

---

## Q: KS only compares one variable's distribution — here, just the target y then why use feature hectares?

**A:** That's exactly the gap the Hectares check fills. KS covers only the target's marginal distribution — it could pass even if feature composition differs between train/test (e.g. train gets more small plots, test more large ones), which would hurt the model but go undetected by KS alone.

Why Hectares specifically (not all 44 columns):
1. Not arbitrary — Hectares is already central to the problem before this check runs (§1.2 defines the sample partly by it "1-6 ha"; domain logic says bigger plot = more yield, before any stats). §8-§10 later confirm formally (r=0.995 with raw yield) but the instinct to check it doesn't need that confirmation first.
2. Only 6 distinct values (1-6 ha) -> a simple % composition table (train vs test) is cheap, exact, transparent — no test statistic needed. Running KS-style tests on all 44 columns would be overkill, especially since §8-§10 later show most columns carry no real signal.
3. Together the two checks cover what one alone can't: KS = "does the outcome's shape match?", Hectares table = "does the composition of the thing that actually drives the outcome match?" Because Hectares and y are tightly linked, a mismatch there would show up as reduced predictive ability across plot sizes even if KS alone looked fine.

So it's a deliberate choice to cover the one feature that turns out to matter, not "check target then a random extra feature."

---

## Q: normally when split test and train. the distribution of train and test must be similar. the distribution of what? target of features of train and test?

**A:** Textbook-correct answer: both, together — the JOINT distribution of features AND target, P(X,y), not just one or the other.

Why joint: the formal assumption behind train/test evaluation is that train and test are both random draws from the same P(X,y). A model learns the X->y relationship; if that relationship or the mix of X differs between train/test, test score stops fairly estimating real-world performance.

Three named failure types (ML literature):
- Covariate shift: distribution of X (features) differs. E.g. train gets more small plots, test more large.
- Label shift: distribution of y (target) differs. E.g. train gets more low-yield rows overall.
- Concept drift: the relationship P(y|X) itself differs. E.g. same Hectares value predicts different yield in train vs test.

Why the notebook checks only target + one feature, not the full joint: literally checking a 45-dim joint distribution match isn't feasible with a simple test. Practical proxy used instead: KS test on target's marginal (catches label shift) + Hectares composition table (catches the covariate shift that would actually matter, since Hectares is the dominant predictor). Not a complete guarantee (can't rule out shift in a minor predictor) but covers what would actually bias the reported result.

Direct answer: ideally joint distribution of target+features (formal requirement); in practice, check target's distribution + the distribution/composition of features believed to matter most — exactly the KS + Hectares pairing in §3.

---

## Q: (pasted Hectares share % table, train vs test by plot size 1-6) what is the meaning of this?

**A:** This is §3's Hectares composition check — the covariate-shift check on the dominant feature, paired with the KS test on the target.

How to read it: each row = train or test; each column = plot size (1-6 ha); each cell = "% of that part's rows that are this plot size." Each row sums to ~100%.

What it checks: whether the plot-size mix is similar in train vs test. If one part got disproportionately more large/small plots, evaluation would be skewed by that imbalance rather than reflecting real model quality.

Actual numbers (train% / test% / gap): 1ha: 8.9/8.3/0.6, 2ha: 14.5/17.1/2.6, 3ha: 16.9/12.7/4.2 (biggest gap), 4ha: 22.7/23.1/0.4, 5ha: 28.5/31.2/2.7, 6ha: 8.5/7.7/0.8. All gaps small, consistent with ordinary random sampling noise (~700 rows out of ~2,300), not systematic bias — basis for §3's "Hectares mix is similar" conclusion.

Why it exists alongside KS: KS checked only the target's distribution; this table separately checks Hectares (the feature y depends on almost entirely), so both halves of "is the split representative" are verified, not assumed.

---

## Q: my understanding is we recheck the percent present of hectares b/w train and test dataset right?

**A:** Correct. For each plot size (1-6 ha), compute what % of train is that size and what % of test is that size, then compare side by side. If the percentages roughly match at every plot size, the split didn't accidentally skew one part toward bigger/smaller plots than the other — important because plot size drives almost all of the yield signal (§8-§10). It's a cross-check of Hectares's composition between train/test, done alongside the KS test on the target, so both "does the outcome distribution match?" and "does the dominant feature's distribution match?" are verified rather than assumed.

---

## Q: (pasted §3 code: rs_tr/rs_te counterfactual split + merge/indicator duplicate-leakage checks) describe code line by line

**A:** Two-part demonstration measuring duplicate-related leakage risk, hypothetically and actually.

**Part 1 — counterfactual "what if duplicates weren't removed":**
- `rs_tr, rs_te = train_test_split(raw, test_size=0.30, random_state=SEED)` — re-splits the ORIGINAL un-deduplicated `raw` (2,789 rows), same seed/ratio as real split. Simulation only, not used elsewhere.
- `hit = rs_te.merge(rs_tr.drop_duplicates(), how='left', indicator=True)['_merge'].eq('both').mean()`:
  - `rs_tr.drop_duplicates()` collapses dupes within rs_tr first.
  - `.merge(..., how='left', indicator=True)` with no `on=` matches on ALL shared columns (all 45) — exact row-for-row match. `_merge` column = 'left_only' (no match) or 'both' (exact match found).
  - `.eq('both')` -> True/False per test row; `.mean()` -> proportion matched (True=1,False=0 averaged).
- print: hit as %. Result: 25.8% of test rows would be exact copies of a train row if dedup hadn't happened first.

**Part 2 — real (already-deduplicated) split, features only:**
- `tw = test[feat_cols].merge(train[feat_cols].drop_duplicates(), how='left', indicator=True)['_merge'].eq('both').mean()` — same merge/indicator trick but on the REAL train/test, restricted to `feat_cols` (44 predictors, target Y excluded). Asks: does the same input combination (ignoring yield) appear in both train and test?
- print: tw as %. Result: 96.2% of test rows share their whole feature vector with some training row (different yield).

Interpretation: Part 2 isn't leakage in the "cheating" sense — it's the already-known fact of only 411 distinct feature combos for 2,789 rows (repetitive inputs, §2). Flagged because a plain random split doesn't really test on "new" input combos — motivates the leave-one-agri-block-out CV in §14.4 as a tougher check.


## Q: what features did you not consider(remove) when you went through each section?
**A:** Full funnel from 44 candidate predictors down to the final 2 (`Hectares`, `Variety`), with the exact section and evidence for each removal:
- `Trash` (1) — §1.4 domain leakage + §8.3 empirical (partial corr flips to −0.16 once Hectares controlled for; −0.03/p=0.19 vs size-adjusted target).
- 10 input columns (Seedrate, LP_Mainfield, Nursery area, LP_nurseryarea, DAP_20days, Weed28D_thiobencarb, Urea_40Days, Potassh_50Days, Micronutrients_70Days, Pest_60Day) — passed §8's filter tests (r=0.9946, same as Hectares) but §10.1 showed each is an exact per-hectare-dose multiple of Hectares; dropped as duplicates.
- 24 weather columns (Rain×4, AI×4, Min/Max temp×4 each, Wind speed×4, RH×4) — no signal in §8.3 (|r|≤0.05, ns); §10.1 showed they're deterministic functions of Agriblock (rank 6, all VIF infinite).
- 4 Wind Direction columns — no signal in §9.3 (p>0.19); §10.1 showed Cramér's V=1.0 vs Agriblock (exact relabelling).
- Agriblock, Soil Types, Nursery (1 each) — no signal in §9.3 (p>0.19 for all); Agriblock additionally excluded to avoid group leakage (§1.4 caution 4 / §14.4 leave-one-block-out check).
- Kept: Hectares (dominant, r≈0.995, survives §8-§11 incl. unsupervised PCA) + Variety (small but Bonferroni-significant, η²=0.008).
- Count check: 1+10+24+4+1+1+1 = 42 removed + 2 kept = 44. ✓
- Also mentioned: §11's 5 domain-engineered aggregates and PCA components were constructed but not kept (no signal); a "wide" model with Agriblock/Soil/Nursery added back was still explicitly fitted in §14 (M2_wide) as an empirical double-check, and it slightly *underperformed* the narrow model out-of-sample, confirming the §9/§10 decision rather than just asserting it.

---
## Q: create copy of src/paddy_ML_regression.ipynb but translate into Thai (specific words still prefer English)
**A:** สร้าง `src/paddy_ML_regression_TH.ipynb` — สำเนาของ notebook เดิมที่แปล markdown ทั้ง 50 cells และ comment ในโค้ด 28 บรรทัดเป็นภาษาไทย โดยคงศัพท์เทคนิคเป็นภาษาอังกฤษ (linear regression, target, feature, leakage, outlier, skewness, heteroscedasticity, ANOVA, PCA, VIF, RMSE/MAPE ฯลฯ) ชื่อคอลัมน์ สูตรคณิตศาสตร์ โค้ด และ outputs ทั้ง 87 ชิ้นไม่ถูกแตะต้อง ไฟล์เดิมไม่เปลี่ยนแปลง

## Q: in section 1 you say we remove Trash but later in section 5 why Trash still live?
**A:** Good catch — `Trash` was never actually removed from the code's working column list, only flagged. §1.4 excludes it by *domain reasoning* (decision: "EXCLUDE (leakage)"), but the code's `NUM` list (defined in §2, comment: "incl. Trash, kept for the audit only") deliberately keeps it in every subsequent EDA/statistical pass — §5's overview table, §5's 36-panel histogram grid, and §8's correlation/MI tests. This was intentional (it's how §8.3 empirically proves the §1.4 leakage call was right: Trash's r=0.958 with raw yield collapses to partial r=−0.16 once Hectares is controlled for, and to r=−0.03/p=0.19 against the size-adjusted target) — but the notebook never said so clearly in the markdown a reader actually sees, only in a code comment. So it looked like Trash was still an active candidate.

**Fix applied (markdown-only, no computation changed, no re-execution needed):**
1. §5 header — added an explicit "**Note on `Trash`**" paragraph, up front, before any code: already excluded in §1.4, kept in tables/figures only as a live audit check, never a candidate from here on.
2. §5.1 discussion — reworded "44 predictors" (which wrongly implied Trash was a live candidate) to "44 non-target columns... but only 43 are real candidates"; reworded the "Trash 17" aside and the "only column rejected up-front" sentence to make the audit-only status explicit at each mention.
3. §8 header — added the same one-line clarification right where Trash's statistical tests actually run.
4. §18 conclusion — fixed "reduced 44 candidate columns to two" (wrong — Trash was never a real candidate) to "reduced 43 real candidate columns to two — plus `Trash`, carried along only to empirically confirm its earlier leakage exclusion".

Verified notebook still 99 cells, valid JSON. Synced all four edits into the source pipeline scripts (p2.py, p3.py, p5.py) that build the notebook.

## Q: also check section 2, 4 and 4 too (for the same "declared vs. actual" type inconsistency found with Trash)
**A:** Interpreted as sections 2, 3, 4 (assumed typo). Verified against actual code/output, not just re-reading prose:
- §2: `df = raw.drop_duplicates().copy()` genuinely drops duplicates; train/test (§3) built from `df` not `raw`. Cross-checked every number in the discussion cells against real cell output: 451 rows removed (2,789→2,338) = 16.17% ≈ §18's "16.2%" ✓; "812 rows / 361 distinct duplicated records / group sizes {2:293,3:50,4:14,5:4}" ✓ exact match.
- §3: KS D=0.0308→"0.031", p=0.724→"0.72" ✓; Hectares share "3 ha: 16.9% vs 12.7%" ✓; "25.8%"/"96.2%" duplicate-leakage figures ✓ — all match printed output exactly.
- §4: "rows per plot size 145/237/277/371/467/139" ✓ matches `tr.Hectares.value_counts()` exactly. The one forward-deferred claim ("largest |z|≈1.9", stated as computed in §6) is honestly flagged as deferred, not silently misstated.
- Verdict: no `Trash`-style bugs (claim in prose vs. different actual code behavior) found in any of these three sections.

## Q: update the src/paddy_ML_regression_TH.ipynb (sync the Trash-consistency fixes)
**A:** ตรวจสอบแล้วว่ามีเพียง 4 markdown cells ที่เปลี่ยนในไฟล์ EN หลังจากสร้าง TH copy (cell 33 §5 header, 37 §5.1, 48 §8 header, 98 §18) — code cells และ outputs ไม่เปลี่ยนเลย จึง patch เฉพาะ 4 cells นั้นในฉบับไทย:
1. §5 header — เพิ่มย่อหน้า "**หมายเหตุเรื่อง `Trash`**": ถูกตัดออกในฐานะ leakage แล้วใน §1.4 คงไว้ในตาราง/รูปเพื่อการ audit เท่านั้น ไม่ใช่ candidate
2. §5.1 — แก้ "44 predictors" เป็น "44 คอลัมน์ที่ไม่ใช่ target ... แต่มีเพียง 43 ตัวที่เป็น candidate จริง" + ระบุสถานะ audit-only ทุกจุดที่กล่าวถึง Trash
3. §8 header — เพิ่มบรรทัดชี้แจงเดียวกัน ตรงจุดที่การทดสอบทางสถิติของ Trash รันจริง
4. §18 — แก้ "ลด 44 คอลัมน์ candidate เหลือ 2" เป็น "ลดคอลัมน์ candidate จริง 43 ตัวเหลือ 2 ตัว — บวกกับ `Trash` ที่พาไปด้วยเพียงเพื่อยืนยันเชิงประจักษ์..."

ยืนยัน: 99 cells, valid JSON, code เหมือน EN ทุกบรรทัด (ยกเว้น comment ที่แปลไทย), outputs ครบ 87 ชิ้น
