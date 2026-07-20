// background.js - Talk Through Me service worker.
// Owns the offscreen document (where camera + ML run) and relays its
// prediction stream to content scripts.

chrome.runtime.onInstalled.addListener(() => {
  console.log("Talk Through Me extension successfully installed.");
  chrome.storage.local.set({
    extensionEnabled: false,
    chatInjectionEnabled: false,
    detectionThreshold: 0.90,
    spacingDelay: 750
  });
});

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA"],
    justification: "Run webcam hand tracking and sign-language recognition"
  });
}

async function closeOffscreen() {
  if (await chrome.offscreen.hasDocument()) {
    await chrome.offscreen.closeDocument();
  }
}

// Restart the engine: tear down the failed offscreen doc and recreate it so it
// retries getUserMedia — used right after the camera permission is granted.
async function restartOffscreen() {
  await closeOffscreen();
  await ensureOffscreen();
}

// Open the one-time camera-grant page, throttled so a burst of "camera-denied"
// frames (or a service-worker restart) can't spam duplicate tabs. Uses storage
// instead of chrome.tabs.query({url}) because that needs the "tabs" permission,
// which this extension intentionally does not request.
async function openPermissionTabOnce() {
  const { lastPermissionTabAt = 0 } = await chrome.storage.local.get("lastPermissionTabAt");
  if (Date.now() - lastPermissionTabAt < 10000) return; // one per 10s max
  await chrome.storage.local.set({ lastPermissionTabAt: Date.now() });
  chrome.tabs.create({ url: chrome.runtime.getURL("permission.html") });
}

function broadcastToTabs(message) {
  // NOTE: no URL filtering — without the "tabs" permission, tab.url is
  // always undefined, so a URL check silently matches nothing. Send to
  // every tab; tabs without our content script reject harmlessly.
  chrome.tabs.query({}, (tabs) => {
    for (const tab of tabs) {
      chrome.tabs.sendMessage(tab.id, message).catch(() => {});
    }
  });
}

// Settings changes: relay to content scripts + manage offscreen lifecycle
chrome.storage.onChanged.addListener((changes, namespace) => {
  if (namespace !== "local") return;

  broadcastToTabs({ action: "settingsChanged", changes });

  if (changes.extensionEnabled) {
    if (changes.extensionEnabled.newValue) {
      ensureOffscreen();
    } else {
      closeOffscreen();
    }
  }
});

// Messages from the offscreen document (frames/status) -> content scripts.
// Messages from content scripts (ensure-offscreen on page load) -> offscreen.
chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || typeof msg.type !== "string") return;

  if (msg.type === "s2tFrame" || msg.type === "s2tStatus") {
    broadcastToTabs(msg);

    // Camera denied: open the one-time permission page (deduped).
    if (msg.type === "s2tStatus" && msg.code === "camera-denied") {
      openPermissionTabOnce();
    }
  }

  // Camera just granted on the permission page: restart the engine so it
  // retries the camera automatically — no manual toggle off/on needed.
  if (msg.type === "s2tPermissionGranted") {
    restartOffscreen();
  }

  if (msg.type === "s2tEnsureOffscreen") {
    ensureOffscreen();
  }
});
