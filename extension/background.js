// background.js - SignToText extension service worker

chrome.runtime.onInstalled.addListener(() => {
  console.log("SignToText extension successfully installed.");
  // Default settings
  chrome.storage.local.set({
    extensionEnabled: false,
    chatInjectionEnabled: false,
    detectionThreshold: 0.90,
    spacingDelay: 750
  });
});

// Broadcast changes in storage to active tabs
chrome.storage.onChanged.addListener((changes, namespace) => {
  if (namespace === "local") {
    chrome.tabs.query({}, (tabs) => {
      for (const tab of tabs) {
        // Forward state change to content script
        if (tab.url && (tab.url.includes("meet.google.com") || tab.url.includes("zoom.us") || tab.url.includes("teams"))) {
          chrome.tabs.sendMessage(tab.id, {
            action: "settingsChanged",
            changes: changes
          }).catch(err => {
            // Content script not loaded on this tab yet, safe to ignore
          });
        }
      }
    });
  }
});
