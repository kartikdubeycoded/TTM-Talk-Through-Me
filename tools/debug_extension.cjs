// Drives Brave with the extension loaded + a FAKE webcam, navigates to
// meet.google.com, force-enables SignToText, and captures every console
// message/error so we can see exactly where init dies.
//
// Run: node tools/debug_extension.cjs
const path = require("path");
const puppeteer = require("puppeteer-core");

const BRAVE = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe";
const EXT = path.resolve(__dirname, "..", "extension");
const VIDEO = path.resolve(__dirname, "..", "test_signs.y4m"); // the ASL clip, fed as the webcam
const fs = require("fs");

(async () => {
  const browser = await puppeteer.launch({
    executablePath: BRAVE,
    headless: false,
    args: [
      `--disable-extensions-except=${EXT}`,
      `--load-extension=${EXT}`,
      "--use-fake-device-for-media-stream",          // pretend a camera exists
      `--use-file-for-fake-video-capture=${VIDEO}`,  // ...and it's our ASL clip
      "--use-fake-ui-for-media-stream",              // auto-grant camera permission
      "--no-first-run",
      "--disable-features=DialMediaRouteProvider",
    ],
  });

  try {
    // Tap the console of every extension page that appears (esp. offscreen.html)
    browser.on("targetcreated", async (t) => {
      if (!t.url().includes("offscreen.html")) return;
      console.log("[harness] offscreen document created");
      try {
        const session = await t.createCDPSession();
        await session.send("Runtime.enable");
        session.on("Runtime.consoleAPICalled", (e) => {
          const text = e.args.map(a => a.value ?? a.description ?? "").join(" ");
          console.log(`[offscreen:${e.type}] ${text}`);
        });
        session.on("Runtime.exceptionThrown", (e) => {
          console.log(`[offscreen:EXCEPTION] ${e.exceptionDetails.text} ` +
            (e.exceptionDetails.exception?.description || ""));
        });
      } catch (err) {
        console.log(`[harness] could not attach to offscreen: ${err.message}`);
      }
    });

    // Find the extension's service worker to get its ID
    const swTarget = await browser.waitForTarget(
      t => t.type() === "service_worker" && t.url().includes("background.js"),
      { timeout: 15000 }
    );
    const extId = new URL(swTarget.url()).host;
    console.log(`[harness] extension loaded, id=${extId}`);

    // Open Meet landing page FIRST (mirrors Katti's real flow) and log everything
    const page = await browser.newPage();
    page.on("console", msg =>
      console.log(`[page:${msg.type()}] ${msg.text()}`));
    page.on("pageerror", err =>
      console.log(`[pageerror] ${err.message}`));

    await page.goto("https://teams.live.com/", { waitUntil: "domcontentloaded" });
    console.log(`[harness] page loaded, final URL: ${page.url()}`);
    console.log("[harness] waiting 5s before toggling on...");
    await new Promise(r => setTimeout(r, 5000));

    // NOW flip the toggle via storage (like clicking the popup switch);
    // background.js broadcasts the change to the open Meet tab.
    const popup = await browser.newPage();
    await popup.goto(`chrome-extension://${extId}/popup.html`);
    const readback = await popup.evaluate(() =>
      new Promise(res => chrome.storage.local.set({ extensionEnabled: true },
        () => chrome.storage.local.get(["extensionEnabled"], res)))
    );
    console.log(`[harness] toggle flipped, storage readback:`, JSON.stringify(readback));

    // Diagnostic: open offscreen.html as a normal tab so we can see its
    // console and inspect the tf global directly.
    const diag = await browser.newPage();
    diag.on("console", msg => console.log(`[offdiag:${msg.type()}] ${msg.text()}`));
    diag.on("pageerror", err => console.log(`[offdiag:pageerror] ${err.message}`));
    await diag.goto(`chrome-extension://${extId}/offscreen.html`);
    await new Promise(r => setTimeout(r, 8000));
    const tfState = await diag.evaluate(() => ({
      tfType: typeof tf,
      hasSequential: typeof tf !== "undefined" && typeof tf.sequential,
      tfKeys: typeof tf !== "undefined" ? Object.keys(tf).slice(0, 15) : null,
      tfVersion: typeof tf !== "undefined" && tf.version ? JSON.stringify(tf.version) : null,
    }));
    console.log(`[harness] tf diagnostic:`, JSON.stringify(tfState));

    // Watch the overlay while the ASL clip plays. Sample the live "Model sees"
    // guess twice a second; log a line only when the guess CHANGES (so the log
    // reads as one line per distinct sign, not a wall of repeats).
    const SECONDS = Number(process.env.AUDIT_SECONDS) || 150;  // watch window
    const rows = [];
    let lastGuess = null;
    for (let i = 0; i < SECONDS * 2; i++) {
      await new Promise(r => setTimeout(r, 500));
      const snap = await page.evaluate(() => {
        const g = document.getElementById("s2t-guess");
        const dot = document.getElementById("s2t-status-text");
        const sentence = document.getElementById("s2t-sentence-container");
        return {
          guess: g ? g.textContent.trim() : null,
          status: dot ? dot.textContent : null,
          sentence: sentence ? sentence.textContent.slice(0, 300) : null,
        };
      });
      const t = ((i + 1) / 2).toFixed(1);
      if (snap && snap.guess && snap.guess !== lastGuess) {
        lastGuess = snap.guess;
        console.log(`[t+${t}s] sees="${snap.guess}"  status="${snap.status}"`);
        rows.push({ t, guess: snap.guess, status: snap.status, sentence: snap.sentence });
      }
    }
    const outPath = path.resolve(__dirname, "..", "test_run_log.json");
    fs.writeFileSync(outPath, JSON.stringify(rows, null, 2));
    const finalSentence = rows.length ? rows[rows.length - 1].sentence : "(none)";
    console.log(`[harness] captured ${rows.length} guess-changes -> ${outPath}`);
    console.log(`[harness] final caption text: "${finalSentence}"`);
  } finally {
    await browser.close();
    console.log("[harness] done");
  }
})();
