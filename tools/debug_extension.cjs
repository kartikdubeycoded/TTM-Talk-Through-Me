// Drives Brave with the extension loaded + a FAKE webcam, navigates to
// meet.google.com, force-enables SignToText, and captures every console
// message/error so we can see exactly where init dies.
//
// Run: node tools/debug_extension.cjs
const path = require("path");
const puppeteer = require("puppeteer-core");

const BRAVE = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe";
const EXT = path.resolve(__dirname, "..", "extension");

(async () => {
  const browser = await puppeteer.launch({
    executablePath: BRAVE,
    headless: false,
    args: [
      `--disable-extensions-except=${EXT}`,
      `--load-extension=${EXT}`,
      "--use-fake-device-for-media-stream", // synthetic camera, no hardware
      "--use-fake-ui-for-media-stream",     // auto-grant camera permission
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

    // Poll the overlay's status every 5s
    for (let i = 0; i < 5; i++) {
      await new Promise(r => setTimeout(r, 5000));
      const status = await page.evaluate(() => {
        const dot = document.getElementById("s2t-status-text");
        const sentence = document.getElementById("s2t-sentence-container");
        return {
          overlayExists: !!document.getElementById("signtotext-overlay"),
          status: dot ? dot.textContent : null,
          sentenceArea: sentence ? sentence.textContent.slice(0, 200) : null,
        };
      });
      console.log(`[harness] t+${(i + 1) * 5}s overlay=${status.overlayExists} ` +
                  `status="${status.status}" text="${status.sentenceArea}"`);
    }
  } finally {
    await browser.close();
    console.log("[harness] done");
  }
})();
