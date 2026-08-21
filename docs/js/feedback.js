// CHIMERA-Agent Shared Non-Blocking Feedback
//
// Standalone toast utility extracted from app.js showFeedback so that static
// view methods (which have no access to the app controller instance) can emit
// transient, non-blocking clinical-workflow messages without resorting to
// synchronous alert() calls that freeze the UI thread.
//
// Exports:
//   showFeedback(message, type = 'info') → void
//
// Lazily creates a shared #app-feedback host element, applies a CSS class per
// severity, and auto-dismisses after ~3 seconds. Mirrors the app.js pattern
// exactly so UX stays consistent across the dashboard.

let _feedbackTimer = null;

export function showFeedback(message, type = 'info') {
  let host = document.getElementById('app-feedback');
  if (!host) {
    host = document.createElement('div');
    host.id = 'app-feedback';
    host.style.position = 'fixed';
    host.style.bottom = '12px';
    host.style.right = '12px';
    host.style.zIndex = '9999';
    host.style.fontFamily = 'var(--font-mono)';
    host.style.fontSize = '12px';
    host.style.padding = '8px 12px';
    host.style.borderRadius = '4px';
    host.style.maxWidth = '360px';
    host.style.opacity = '0';
    host.style.transition = 'opacity 0.3s ease';
    host.style.pointerEvents = 'none';
    document.body.appendChild(host);
  }

  host.textContent = message;
  host.className = `feedback-${type}`;
  host.style.opacity = '1';

  if (_feedbackTimer) clearTimeout(_feedbackTimer);
  _feedbackTimer = setTimeout(() => {
    host.style.opacity = '0';
  }, 3000);
}
