// background.js - SignToText service worker.
// Owns the offscreen document (where camera + ML run) and relays its
// prediction stream to content scripts.

chrome.runtime.onInstalled.addListener(() => {
  console.log("SignToText extension successfully installed.");
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
let permissionTabOpened = false;
chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || typeof msg.type !== "string") return;

  if (msg.type === "s2tFrame" || msg.type === "s2tStatus") {
    broadcastToTabs(msg);

    // First camera denial: open the one-time permission page
    if (msg.type === "s2tStatus" && msg.code === "camera-denied" && !permissionTabOpened) {
      permissionTabOpened = true;
      chrome.tabs.create({ url: chrome.runtime.getURL("permission.html") });
    }
  }

  if (msg.type === "s2tEnsureOffscreen") {
    ensureOffscreen();
  }
});
