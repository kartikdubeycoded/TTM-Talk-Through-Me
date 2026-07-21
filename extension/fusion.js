// fusion.js — the referee between the two recognizers (T18).
//
// TTM runs two models at once:
//   • a static ALPHABET classifier (fingerspelling, letter by letter), and
//   • a temporal WORD LSTM (whole-word signs).
// They fight. The word LSTM sees the hand moving between letters and can fire a
// spurious whole word right in the middle of someone deliberately spelling a
// name (K-A-T-T-I → "kite"). This module holds the precedence rule that stops
// that, in one pure, testable place. content.js owns the timing + state; it just
// asks this function "given where we are, may a word sign speak right now?".
//
// The rule, in words: while the user is actively fingerspelling — a letter
// buffer is open, or a letter locked in very recently — the word model is
// muzzled. Only an *overwhelmingly* confident word sign may interrupt a spell.
// Once spelling has clearly stopped (buffer empty AND quiet for a beat), the
// word model speaks freely at its normal threshold.

// How long after the last locked letter we still treat the user as "mid-spell".
export const FINGERSPELL_LOCKOUT_MS = 1500;

// The only confidence that lets a word sign interrupt active fingerspelling.
// Deliberately high — interrupting a spell is expensive, so demand near-certainty.
export const WORD_OVERRIDE_CONFIDENCE = 0.97;

/**
 * Decide whether a whole-word sign may commit right now.
 *
 * @param {object} s
 * @param {number} s.bufferLength      length of the in-progress fingerspell buffer
 * @param {number} s.msSinceLastLetter ms since the last letter locked in
 * @param {number} s.wordConfidence    the word LSTM's confidence for this sign
 * @returns {boolean} true if the word sign is allowed to inject
 */
export function wordSignAllowed({ bufferLength, msSinceLastLetter, wordConfidence }) {
  const midFingerspelling =
    bufferLength > 0 || msSinceLastLetter < FINGERSPELL_LOCKOUT_MS;
  if (!midFingerspelling) return true; // not spelling → normal word threshold applies
  return wordConfidence >= WORD_OVERRIDE_CONFIDENCE; // spelling → only a near-certain word wins
}
