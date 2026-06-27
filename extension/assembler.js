// assembler.js — turns the recognition token stream into readable sentences.
// Pure and DOM-free so tests/smoke_assembler.mjs can verify it; content.js owns
// the timing (deciding when a pause ends a sentence) and the rendering.
//
// Two kinds of token come in:
//   addSpelledWord(w) — a fingerspelled buffer (letter by letter). Run through a
//                       conservative autocorrect, since a single wrong letter is
//                       common.
//   addSignedWord(w)  — a whole-word sign from the model's trusted vocabulary.
//                       Never autocorrected (e.g. "dad" must not become "bad").

const QUESTION_WORDS = new Set([
  "who", "what", "where", "when", "why", "how", "which",
]);

// Small, conservative dictionary for fingerspell autocorrect: common words plus
// the kind of words a Deaf user fingerspells. Kept short on purpose — a bigger
// dictionary corrects more aggressively and risks rewriting names into junk.
const DICTIONARY = [
  "the", "and", "you", "are", "for", "not", "but", "with", "this", "that",
  "have", "your", "here", "there", "please", "thank", "what", "where", "when",
  "why", "how", "good", "name", "help", "need", "want", "sorry", "fine",
  "food", "water", "home", "work", "today", "hello", "later", "morning",
];

function editDistance(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

// Fix a fingerspelled word only when it is unambiguously one edit from a single
// dictionary word. Short words (<3) are left alone — too easy to "correct" into
// the wrong thing.
export function autocorrect(word) {
  const w = word.toLowerCase();
  if (w.length < 3) return word;
  if (DICTIONARY.includes(w)) return w;
  const near = DICTIONARY.filter(
    (d) => Math.abs(d.length - w.length) <= 1 && editDistance(w, d) <= 1
  );
  return near.length === 1 ? near[0] : word;
}

export function createAssembler() {
  let sentences = []; // finished, rendered sentences (with punctuation)
  let words = [];      // words in the current, in-progress sentence

  function renderSentence(ws, terminal) {
    let text = ws.join(" ");
    text = text.charAt(0).toUpperCase() + text.slice(1);
    if (terminal) text += QUESTION_WORDS.has(ws[0]) ? "?" : ".";
    return text;
  }

  function render() {
    const parts = sentences.slice();
    if (words.length > 0) parts.push(renderSentence(words, false));
    return parts.join(" ");
  }

  return {
    addSpelledWord(word) {
      const w = autocorrect(word).toLowerCase();
      if (w) words.push(w);
    },
    addSignedWord(word) {
      if (word) words.push(word.toLowerCase()); // trusted vocab — no autocorrect
    },
    endSentence() {
      if (words.length === 0) return;
      sentences.push(renderSentence(words, true));
      words = [];
    },
    backspace() {
      if (words.length > 0) words.pop();
      else if (sentences.length > 0) sentences.pop();
    },
    clear() {
      sentences = [];
      words = [];
    },
    get text() {
      return render();
    },
  };
}
