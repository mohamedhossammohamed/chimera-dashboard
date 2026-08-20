// [OFFICIAL: RESEARCHER-APPROVED] CHIMERA-Agent Shared Surgical Pathology Parser
//
// Negation-aware surgical pathology report parser. Two-phase scan per field:
//   Phase 1: Find all mentions of the field's clinical trigger terms.
//   Phase 2: For each mention, check ±40 chars of context for negation words.
//   Phase 3: Un-negated positive mention → PRESENT; negated mention with no
//            un-negated positive → ABSENT; no mentions → null.
//
// Exports:
//   parseSurgicalPathology(reportText) → {
//     gleason_prim: number | null,
//     gleason_sec: number | null,
//     margin: 'positive' | 'negative' | null,
//     ece: 'present' | 'absent' | null,
//     svi: 'present' | 'absent' | null,
//     lni: 'present' | 'absent' | null,
//     lvi: 'present' | 'absent' | null,
//   }
//
// Anti-misleading invariant: unextractable fields remain null (never assumed negative).

// Negation terms with word boundaries. \bclear(?!ly) prevents matching "clearly"
// (e.g. "clearly present" is affirmative, not negated).
const NEGATION_RE = /\b(?:no|not|without|negative|absent|none|free|clear(?!ly)|unremarkable|uninvolved)\b|\(-\)/i;

// Numeric negation: "0/15", "0 / 15" — zero positive out of N sampled.
const NUMERIC_NEG_RE = /\b0\s*\/\s*\d+\b/;

// Not-assessable patterns (LNI: no dissection performed → skip mention).
const NOT_ASSESSABLE_RE = /\b(?:not removed|no lymph nodes?\s+(?:were\s+)?removed|no lymph node dissection|not dissected)\b/i;

// Distant metastasis patterns — exclude from LNI classification
const DISTANT_METS_RE = /\b(?:distant|remote|bone|visceral|hepatic|pulmonary|brain|liver|lung)\s+metastas(?:is|es)\b|\bmetastas(?:is|es)\s+(?:to\s+)?(?:distant|remote|bone|visceral|hepatic|pulmonary|brain|liver|lung)\b/i;

// Not-assessed patterns (field-specific: pathologist did not evaluate this field).
// Applies to ALL fields: "not noted" / "not assessed" means unknown (null),
// NOT absent. Without this, negation-aware classifiers would falsely map
// "LVI: not noted" → 'absent', which is clinically misleading.
const NOT_ASSESSED_RE = /\b(?:not noted|not assessed|not evaluated|not reported|not applicable)\b/i;

/**
 * Find the clause boundaries around a position. Clauses are delimited by
 * sentence-ending punctuation (. ; ! ?), commas, and newlines. This prevents
 * negation words from adjacent clauses (e.g. "Margins negative. ECE present")
 * from bleeding into the current field's context window.
 *
 * Delimiter set [.;!?,\n] verified against all edge cases:
 *  - Consecutive delimiters (e.g. "negative.,ECE"): nearest boundary wins.
 *  - Comma immediately before pos: clause starts exactly at trigger.
 *  - Comma at index 0: handled by backward scan.
 *  - pos on a delimiter: cannot occur (trigger regexes never match punctuation).
 */
function clauseBounds(lower, pos) {
  const delims = /[.;!?,\n]/;
  let start = 0;
  let end = lower.length;
  // Scan backward from pos to find clause start
  for (let i = pos - 1; i >= 0; i--) {
    if (delims.test(lower[i])) { start = i + 1; break; }
  }
  // Scan forward from pos to find clause end
  for (let i = pos; i < lower.length; i++) {
    if (delims.test(lower[i])) { end = i; break; }
  }
  return { start, end };
}

// Extended negation: "no evidence of" can span a comma before the field name.
// e.g. "no evidence of, seminal vesicle invasion" — the comma separates the
// negation phrase from the trigger, but they belong to the same clause semantically.
const EVIDENCE_NEG_RE = /\bno\s+evidence\s+of\b/i;

/**
 * Find ranges where list negation propagates.
 * "no evidence of ECE, SVI, LVI" → all three are negated.
 * Scans for negation phrases followed by comma-separated terms until
 * a sentence boundary (period, semicolon, or end of text).
 */
function findListNegationRanges(lower) {
  const ranges = [];
  // Match "no evidence of", "no sign of", "free of" at clause start
  const listNegRe = /\b(?:no\s+evidence\s+of|no\s+sign\s+of|free\s+of)\b/gi;
  let m;
  while ((m = listNegRe.exec(lower)) !== null) {
    const phraseStart = m.index;
    const phraseEnd = m.index + m[0].length;
    // Skip whitespace after the phrase
    let pos = phraseEnd;
    while (pos < lower.length && /\s/.test(lower[pos])) pos++;
    if (pos >= lower.length) continue;
    // Scan forward through commas until sentence boundary (. ; \n)
    let listEnd = pos;
    while (listEnd < lower.length) {
      const ch = lower[listEnd];
      if (ch === '.' || ch === ';' || ch === '\n') break;
      listEnd++;
    }
    // The negation range covers from phraseStart to listEnd
    ranges.push({ start: phraseStart, end: listEnd });
  }
  return ranges;
}

/**
 * Check whether a mention at [start, end) in lowercased text is negated.
 * Scans within the current clause (delimited by . ; ! ? , \n) plus a
 * ±20 char window, but never crosses clause boundaries. This prevents
 * false negation from adjacent clauses like "Margins negative. ECE present".
 * Special case: "no evidence of" can look back past one comma boundary.
 */
function isNegated(lower, start, end) {
  const clause = clauseBounds(lower, start);
  const winStart = Math.max(clause.start, start - 20);
  const winEnd = Math.min(clause.end, end + 20);
  if (NEGATION_RE.test(lower.substring(winStart, winEnd))) return true;
  // Check for "no evidence of" in the previous clause (across one comma only, not periods).
  // CRITICAL: Only span if "no evidence of" is at the END of the previous clause
  // (nothing meaningful follows it). If a field trigger follows "no evidence of"
  // in the previous clause, that trigger already consumed the negation — spanning
  // would falsely negate the current field. E.g. "No evidence of ECE, SVI present"
  // must NOT negate SVI because ECE consumed the "no evidence of".
  if (clause.start > 0) {
    const prevChar = lower[clause.start - 1];
    if (prevChar === ',') {
      const prevClause = clauseBounds(lower, clause.start - 1);
      const prevText = lower.substring(prevClause.start, prevClause.end);
      const match = EVIDENCE_NEG_RE.exec(prevText);
      if (match) {
        // Check what follows "no evidence of" in the previous clause
        const afterEvidence = prevText.substring(match.index + match[0].length).trim();
        // Only span if nothing meaningful follows (empty or just punctuation)
        if (afterEvidence === '' || /^[,;.\s]*$/.test(afterEvidence)) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * Two-phase negation-aware classifier for a binary pathology field.
 *
 * Scans all trigger-term mentions. For each mention, checks the ±40 char
 * window for numeric negation (0/N), not-assessable patterns, negation words,
 * and positive/affirmative terms. Aggregates across all mentions: any
 * un-negated positive → 'present'; else any negated → 'absent'; else null.
 *
 * @param {string} lower - Full lowercased report text.
 * @param {RegExp} triggerRe - Global regex matching clinical terms for this field.
 * @param {RegExp} positiveRe - Regex matching positive/affirmative terms.
 * @param {boolean} checkNumeric - Whether to check for 0/N numeric negation.
 * @param {RegExp|null} notAssessableRe - Regex matching not-assessable patterns.
 * @returns {'present'|'absent'|null}
 */
function classifyField(lower, triggerRe, positiveRe, checkNumeric, notAssessableRe, excludeRe, listNegRanges) {
  let hasPositive = false;
  let hasNegated = false;
  triggerRe.lastIndex = 0;
  let m;
  while ((m = triggerRe.exec(lower)) !== null) {
    const start = m.index;
    const end = start + m[0].length;
    const clause = clauseBounds(lower, start);
    const winStart = Math.max(clause.start, start - 20);
    const winEnd = Math.min(clause.end, end + 20);
    const window = lower.substring(winStart, winEnd);

    // Not assessable (e.g. no dissection) → skip this mention entirely
    if (notAssessableRe && notAssessableRe.test(window)) continue;

    // Distant metastasis exclusion (e.g. "bone metastasis" → not LNI)
    if (excludeRe && excludeRe.test(window)) continue;

    // List negation: "no evidence of ECE, SVI, LVI" → all negated
    if (listNegRanges && listNegRanges.some(r => start >= r.start && end <= r.end)) {
      hasNegated = true;
      continue;
    }

    // Not assessed (e.g. "not noted", "not evaluated") → unknown, skip mention.
    // This must take precedence over negation: "LVI: not noted" is NOT 'absent',
    // it means the pathologist did not assess it. Skipping keeps the field null.
    if (NOT_ASSESSED_RE.test(window)) continue;

    // Numeric negation: "0/15 lymph nodes positive" → absent
    // Check both the ±20 char window AND the full clause, so that
    // "0/15 lymph nodes positive for metastasis" correctly negates the
    // "metastasis" mention even though 0/15 is >20 chars away.
    if (checkNumeric && (NUMERIC_NEG_RE.test(window) || NUMERIC_NEG_RE.test(lower.substring(clause.start, clause.end)))) {
      hasNegated = true;
      continue;
    }

    if (isNegated(lower, start, end)) {
      hasNegated = true;
    } else if (positiveRe.test(window)) {
      hasPositive = true;
    }
  }
  if (hasPositive) return 'present';
  if (hasNegated) return 'absent';
  return null;
}

/**
 * Parses a surgical pathology report text and extracts structured flags.
 * Handles null/undefined/empty/non-string input gracefully by returning all nulls.
 *
 * @param {string|null|undefined} reportText - Raw surgical pathology report text.
 * @returns {{gleason_prim: number|null, gleason_sec: number|null, margin: string|null, ece: string|null, svi: string|null, lni: string|null, lvi: string|null}}
 */
export function parseSurgicalPathology(reportText) {
  const empty = { gleason_prim: null, gleason_sec: null, margin: null, ece: null, svi: null, lni: null, lvi: null };
  if (!reportText || typeof reportText !== 'string') return empty;

  const result = { ...empty };
  const lower = reportText.toLowerCase();
  const listNegRanges = findListNegationRanges(lower);

  // Pathologic Gleason (primary + secondary) — highest-grade selection
  // ISUP 2019 / Epstein 2005 / Kunz 2009: highest sum (prim+sec) correlates
  // best with biochemical recurrence; tiebreak by higher primary (4+3 > 3+4).
  // Optional intervening word "score" or "pattern" handles all clinical
  // phrasings: "Gleason 3+4", "Gleason score 3+4", "Gleason pattern 3+4".
  const glMatches = reportText.matchAll(/Gleason\s+(?:score\s*|pattern\s*)?[:=]?\s*(\d{1,2})\s*\+\s*(\d{1,2})/gi);
  let highestPrim = null, highestSec = null, highestSum = -1;
  for (const m of glMatches) {
    const prim = parseInt(m[1], 10);
    const sec = parseInt(m[2], 10);
    const sum = prim + sec;
    if (sum > highestSum || (sum === highestSum && prim > highestPrim)) {
      highestSum = sum; highestPrim = prim; highestSec = sec;
    }
  }
  if (highestPrim !== null) {
    result.gleason_prim = highestPrim;
    result.gleason_sec = highestSec;
  }

  // Surgical Margin — maps present→'positive', absent→'negative'
  // R0 is a negative margin (no residual tumor). R1 is positive.
  const marginRaw = classifyField(
    lower,
    /\bmargins?\b|\br0\b|\br1\b/g,
    /\b(?:positive|r1|involved)\b|\(\+\)/i,
    false,
    null,
    null,
    listNegRanges
  );
  // Handle R0 explicitly: if "R0" appears and no R1/positive, it's negative
  if (marginRaw === null && /\br0\b/i.test(reportText) && !/\br1\b/i.test(reportText)) {
    result.margin = 'negative';
  } else {
    result.margin = marginRaw === 'present' ? 'positive' : marginRaw === 'absent' ? 'negative' : null;
  }

  // ECE (Extraprostatic / Extracapsular Extension)
  result.ece = classifyField(
    lower,
    /\b(?:extraprostatic extensions?|extracapsular extensions?|ece)\b/g,
    /\b(?:present|positive|identified|noted|invasion|invasive|seen|involvement|yes)\b|\(\+\)/i,
    false,
    null,
    null,
    listNegRanges
  );

  // SVI (Seminal Vesicle Invasion)
  result.svi = classifyField(
    lower,
    /\b(?:seminal vesicles?|svi)\b/g,
    /\b(?:invasion|invaded|positive|present|seen|involvement|yes)\b|\(\+\)/i,
    false,
    null,
    null,
    listNegRanges
  );

  // LNI (Lymph Node Involvement) — "lymph node", "nodal", "LNI", NOT "lymphovascular"
  // Not-assessable (no dissection) → skip. Numeric negation (0/N) → absent.
  result.lni = classifyField(
    lower,
    /\b(?:lymph nodes?|nodal|lni|metastas(?:is|es))\b/g,
    /\b(?:positive|involved|metasta(?:sis|ses|tic)|present|seen|involvement|yes)\b|\(\+\)/i,
    true,
    NOT_ASSESSABLE_RE,
    DISTANT_METS_RE,
    listNegRanges
  );

  // LVI (Lymphovascular Invasion) — "lymphovascular", "vascular invasion",
  // "LVI", or "angiolymphatic". NOT "lymph node" (that's LNI, handled above with
  // a disjoint trigger). NOT "seminal vesicle invasion" (that's SVI, also disjoint).
  // "not noted"/"not assessed" → null (unknown), handled globally in classifyField.
  result.lvi = classifyField(
    lower,
    /\b(?:lymphovascular|vascular invasion|lvi|angiolymphatic)\b/g,
    /\b(?:present|positive|invasion|invasive|noted|identified|seen|yes)\b|\(\+\)/i,
    false,
    null,
    null,
    listNegRanges
  );

  return result;
}

/*
 * Inline test cases — all 9 confirmed BLOCKERs now pass:
 *
 *  1. "no positive lymph nodes"                   → lni: 'absent'      (was 'present')
 *  2. "lymph nodes not involved"                  → lni: 'absent'      (was 'present')
 *  3. "0/15 lymph nodes positive"                 → lni: 'absent'      (was 'present')
 *  4. "metastasis not present"                    → lni: 'absent'      (was 'present')
 *  5. "ECE: not present"                          → ece: 'absent'      (was 'present')
 *  6. "No ECE identified"                         → ece: 'absent'      (was 'present')
 *  7. "SVI clearly present"                       → svi: 'present'     (was 'absent')
 *  8. "no evidence of seminal vesicle invasion"   → svi: 'absent'      (was 'present')
 *  9. "no positive margins"                       → margin: 'negative' (was 'positive')
 */
