# Hiberno-English Transformation Failure Analysis

## Executive Summary

Analysis of Trans-EnV's "Irish" dialect transformations reveals systematic failures in producing authentic Hiberno-English. The transformations apply generic dialect markers (primarily "like" insertion) rather than genuine Hiberno-English syntactic features.

## Sample Analysis

### Examples from `irish_rerun.csv`

| Original | "Transformed" | Issues |
|----------|---------------|--------|
| "The robe takes 2 bolts of blue fiber and half that much white fiber" | "The robe takes 2 bolts of blue fiber and **like**, half that much white fiber" | "Like" insertion is NOT Hiberno-English |
| "She gives them 15 cups of feed" | "She gives them, **like**, 15 cups of feed" | Same issue - California English filler |
| "Kylar went to the store to buy glasses" | "Kylar went to the store **for to** buy glasses" | Correct feature but inconsistent application |
| "How long does it take" | "How **load** does it take" | Typo/corruption, semantic drift |
| "James runs 3 sprints" | "James runs 3 **the** sprints" | Ungrammatical, not authentic |

## Failure Categories

### 1. Wrong Dialect Features Applied
- **"Like" as filler**: Applied extensively but this is California/Valley Girl English, not Hiberno-English
- Generic features from DIALECT_FEATURE_LIST applied without dialect specificity

### 2. Missing Authentic Hiberno-English Features
The following well-documented features are ABSENT from transformations:

| Feature | Example | Linguistic Source |
|---------|---------|-------------------|
| Perfective "after" | "I'm after eating" = "I have just eaten" | Filppula (1999), Hickey (2007) |
| Habitual "do be" | "He does be working late" | Kallen (2013) |
| "Amn't" contraction | "I amn't sure" | Standard in Irish English |
| Plural "youse/ye" | "Are youse coming?" | Second person plural distinction |
| "Sure" as discourse marker | "Sure, wasn't I just saying..." | Pragmatic marker |
| Cleft sentences | "It's tired I am" | Irish substrate influence |
| Reflexive emphasis | "Is himself coming?" | Referring to important person |
| "And" + subject + gerund | "And me sitting there" | Narrative construction |
| Embedded inversion | "I wonder is he coming" | Indirect question structure |

### 3. Semantic Drift
- Some transformations change meaning or introduce errors
- Example: "How load" instead of "How long"

### 4. Inconsistent Application
- "For to" infinitive (authentic feature) applied sporadically
- No systematic application of syntactic transformations

## Root Cause Analysis

### Trans-EnV Architecture Issues

1. **Generic Feature List**: `DIALECT_FEATURE_LIST` in `registry/guidline.py` contains generic features applied to ALL dialects
2. **eWAVE Dependency**: System relies on eWAVE database for dialect-feature mapping, but:
   - eWAVE data not included in repository
   - Feature mapping may be incomplete for Hiberno-English
3. **Guideline Generation**: Guidelines in `orig_generated_guideline_wo_example.json` are LLM-generated and not linguistically validated
4. **No Dialect-Specific Validation**: No mechanism to verify transformations match target dialect

### Code Flow
```
guideline.py:dialect_feature()
  -> Queries eWAVE for features where Value='A' (pervasive)
  -> Filters by DIALECT_FEATURE_LIST
  -> Returns generic features, not Hiberno-English specific
```

## Recommendations

### Immediate Fixes
1. Create Hiberno-English-specific feature specification (`data/dialects/hiberno_english.yaml`)
2. Implement feature detection to verify presence of target dialect markers
3. Add semantic similarity validation to catch meaning drift

### Validation Pipeline Requirements
1. **Semantic Validation**: Reject if similarity < 0.85 with original
2. **Feature Validation**: Require presence of at least N authentic features
3. **Authenticity Validation**: Filter stereotyped "stage Irish" outputs

### Improved Transformation Prompts
Replace generic guidelines with Hiberno-English-specific transformation rules that:
- Include authentic syntactic patterns
- Provide real examples from Irish English corpora
- Avoid stereotypes and "Oirish" markers

## References

- Filppula, M. (1999). *The Grammar of Irish English*
- Hickey, R. (2007). *Irish English: History and Present-Day Forms*
- Kallen, J. (2013). "Irish English" in *The Mouton World Atlas of Variation in English*
- ICE-Ireland Corpus (International Corpus of English - Ireland component)
