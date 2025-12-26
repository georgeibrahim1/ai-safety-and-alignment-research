// Load saved settings
document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.sync.get(['apiKey'], (result) => {
    if (result.apiKey) {
      document.getElementById('apiKey').value = result.apiKey;
    }
  });
});

// Save settings
document.getElementById('settingsForm').addEventListener('submit', (e) => {
  e.preventDefault();
  
  const apiKey = document.getElementById('apiKey').value.trim();
  const statusDiv = document.getElementById('status');
  
  if (!apiKey) {
    showStatus('Please enter an API key', 'error');
    return;
  }
  
  // Basic validation for Gemini API key format
  if (!apiKey.startsWith('AIza')) {
    showStatus('Invalid API key format. Should start with AIza', 'error');
    return;
  }
  
  // Save to storage
  chrome.storage.sync.set({ apiKey }, () => {
    showStatus('Settings saved successfully!', 'success');
    
    // Clear success message after 2 seconds
    setTimeout(() => {
      statusDiv.style.display = 'none';
    }, 2000);
  });
});

function showStatus(message, type) {
  const statusDiv = document.getElementById('status');
  statusDiv.textContent = message;
  statusDiv.className = `status ${type}`;
  statusDiv.style.display = 'block';
}