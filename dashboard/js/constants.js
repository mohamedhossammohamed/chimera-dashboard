// CHIMERA-Agent Shared Constants (constants.js)
// Zero-Dependency Pure ES6 Module — single source of truth for sentinel
// values used across clinical_engine.js, standard_view.js, and
// standard_components.js.
//
// [OFFICIAL: RESEARCHER-APPROVED] Canonical fail-closed sentinel for
// unrecorded/non-classifiable EAU data (KI-8).
// [SUGGESTION: CO-PILOT] Shared module extraction to eliminate duplication.

// Canonical EAU "data not recorded" sentinel. Every module that needs to
// represent unrecorded/non-classifiable clinical data MUST import this
// constant instead of defining a local copy.
export const EAU_SENTINEL = '[DATA NOT RECORDED]';

// All sentinel variants recognised across the codebase. Used by
// isSentinel() and by standard_components.js's EAU scorecard missing-tier
// check. Kept in sync with the Python MISSING constant
// ("[DATA NOT RECORDED]") and the additional display variants
// ("Indeterminate", "Missing", "MISSING", "").
export const SENTINELS = [
  '[DATA NOT RECORDED]',
  'Indeterminate',
  'Missing',
  'MISSING',
  '',
];

// Returns true when `val` is null, undefined, or matches one of the
// SENTINELS after trimming. Centralises the ad-hoc sentinel checks that
// were previously scattered across multiple files.
export function isSentinel(val) {
  if (val === null || val === undefined) return true;
  const s = String(val).trim();
  return SENTINELS.includes(s);
}
