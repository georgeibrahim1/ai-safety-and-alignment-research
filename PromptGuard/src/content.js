let promptGuardButton = null;
let apiKey = '';

chrome.storage.sync.get(['apiKey'], (result) => {
  apiKey = result.apiKey || '';
});

chrome.storage.onChanged.addListener((changes) => {
  if (changes.apiKey) {
    apiKey = changes.apiKey.newValue || '';
  }
});

function getInputElement() {
  const hostname = window.location.hostname;
  
  if (hostname.includes('chatgpt.com')) {
    return document.querySelector('#prompt-textarea') || 
           document.querySelector('textarea[data-id]');
  } else if (hostname.includes('claude.ai')) {
    return document.querySelector('div[contenteditable="true"]') ||
           document.querySelector('textarea');
  } else if (hostname.includes('gemini.google.com')) {
    return document.querySelector('rich-textarea') ||
           document.querySelector('textarea');
  }
  return null;
}

function getInputText(element) {
  if (!element) return '';
  if (element.tagName === 'TEXTAREA') {
    return element.value;
  }
  return element.innerText || element.textContent || '';
}

function createPromptGuardButton() {
  if (promptGuardButton) return;
  
  const button = document.createElement('button');
  button.id = 'promptguard-button';
  button.innerHTML = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      <path d="M9 12l2 2 4-4"/>
    </svg>
    <span>Check Prompt Safety</span>
  `;
  button.className = 'promptguard-btn';
  button.title = 'Click to analyze prompt safety before submitting';
  
  button.addEventListener('click', handlePromptCheck);
  
  document.body.appendChild(button);
  promptGuardButton = button;
}

async function handlePromptCheck(e) {
  e.preventDefault();
  
  if (!apiKey) {
    showNotification('Please set your API key in the extension popup first!', 'error');
    return;
  }
  
  const inputElement = getInputElement();
  const userPrompt = getInputText(inputElement);
  
  if (!userPrompt.trim()) {
    showNotification('No prompt found. Please write something first!', 'warning');
    return;
  }
  
  showNotification('Analyzing prompt...', 'loading');
  
  try {
    const enhancedPrompt = await analyzePrompt(userPrompt);
    const misalignmentPercent = await getMisalignmentScore(userPrompt);
    showResults(userPrompt, enhancedPrompt, misalignmentPercent, inputElement);
    
  } catch (error) {
    showNotification('Error: ' + error.message, 'error');
  }
}

async function analyzePrompt(userPrompt) {
    const systemPrompt = `
You are an AI prompt-alignment assistant.

Your task is to rewrite the user's prompt to improve clarity, safety, and neutrality,
while preserving the user's original intent as closely as possible.

Objectives (in priority order):
1. Preserve the user's intent and requested task.
2. Remove ambiguity and improve clarity.
3. Reduce risk of unsafe, misleading, or biased outputs.
4. Avoid assumptions, exaggeration, or unnecessary confidence.
5. Keep the prompt simple and robust for small language models.

Guidelines:
- Do not refuse or block the prompt unless it is clearly illegal.
- Do not add moral judgments, policy references, or disclaimers.
- Do not introduce new goals or content not implied by the user.
- If information may be uncertain, allow for uncertainty.
- If evaluation or comparison is requested, require clear criteria.
- If people or groups are mentioned, avoid assumptions unless explicitly required.

Output:
Return only the revised prompt text.
Do not include explanations, notes, or formatting.

User prompt: ${userPrompt}`;
//   const systemPrompt = `You are PromptGuard, an AI prompt-alignment assistant.
// Your task is to rewrite user prompts to be safer, more aligned, and less biased,
// while preserving the user's original intent as much as possible.
// Core objectives (in priority order):
// 1. Preserve the user's intent and task.
// 2. Reduce risk of harmful, unsafe, or unethical outputs.
// 3. Minimize demographic and evaluative bias.
// 4. Reduce prompt features that may trigger strategic or context-dependent behavior
//    (e.g., alignment faking).
// 5. Improve clarity, neutrality, and robustness of the prompt.
// Rules:
// - Do NOT refuse or block prompts unless they are explicitly illegal.
// - Do NOT moralize, lecture, or mention policies.
// - Do NOT add unnecessary safety disclaimers.
// - Do NOT introduce new tasks or content not implied by the user.
// - Keep the enhanced prompt concise and practical.
// - If the prompt involves evaluation, ranking, or scoring, require objective criteria and reasoning.
// - If the prompt involves people or groups, avoid demographic assumptions unless explicitly required.
// Output format:
// Return ONLY the enhanced prompt text.
// Do not include explanations or commentary.

// USER INPUT: ${userPrompt}`;

  const response = await fetch(`https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key=${apiKey}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      contents: [{
        parts: [{
          text: systemPrompt
        }]
      }],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1000,
      }
    })
  });
  
  if (!response.ok) {
    const error = await response.json();
    console.error('Gemini API Error:', error);
    throw new Error(error.error?.message || `API request failed: ${response.status}`);
  }
  
  const data = await response.json();
  console.log('Gemini API Response:', data);
  
  if (!data.candidates || data.candidates.length === 0) {
    throw new Error('No response generated. The prompt may have triggered safety filters.');
  }
  
  const candidate = data.candidates[0];

  if (!candidate.content || !candidate.content.parts || candidate.content.parts.length === 0) {
    throw new Error('Invalid API response format');
  }
  
  return candidate.content.parts[0].text.trim();
}

async function getMisalignmentScore(userPrompt) {
  const scorePrompt = `You must respond with ONLY a single number between 0 and 100. Nothing else.

Analyze this prompt for potential misalignment risk as a percentage (0-100%):
- 0-20%: Very safe, minimal risk
- 21-50%: Moderate risk, some concerns
- 51-100%: High risk, significant safety concerns

Consider: harmful outputs, bias, ambiguity, manipulation potential.

PROMPT TO ANALYZE: ${userPrompt}

Remember: Respond with ONLY the number. Example: 25`;

  const response = await fetch(`https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key=${apiKey}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      contents: [{
        parts: [{
          text: scorePrompt
        }]
      }],
      generationConfig: {
        temperature: 0.3,
        maxOutputTokens: 50,
      }
    })
  });
  
  if (!response.ok) {
    throw new Error('Failed to get misalignment score');
  }
  
  const data = await response.json();
  console.log('Gemini Score API Response:', data);
  
  if (!data.candidates || data.candidates.length === 0) {
    console.warn('No score candidates, defaulting to 0');
    return 0;
  }
  
  const candidate = data.candidates[0];
  
  if (!candidate.content || !candidate.content.parts || candidate.content.parts.length === 0) {
    console.warn('No score content, defaulting to 0');
    return 0;
  }
  
  const scoreText = candidate.content.parts[0].text.trim();
  console.log('Score text received:', scoreText);
  
  const score = parseInt(scoreText.match(/\d+/)?.[0] || '0');
  return Math.min(100, Math.max(0, score));
}

function showResults(original, enhanced, misalignmentPercent, inputElement) {
  const existingModal = document.getElementById('promptguard-modal');
  if (existingModal) existingModal.remove();

  const existingNotification = document.getElementById('promptguard-notification');
  if(existingNotification) existingNotification.remove();
  
  const modal = document.createElement('div');
  modal.id = 'promptguard-modal';
  modal.innerHTML = `
    <div class="promptguard-modal-content">
      <div class="promptguard-modal-header">
        <h2>PromptGuard Analysis</h2>
        <button class="promptguard-close">&times;</button>
      </div>
      
      <div class="promptguard-score">
        <div class="score-label">Misalignment Risk</div>
        <div class="score-value ${getScoreClass(misalignmentPercent)}">${misalignmentPercent}%</div>
        <div class="score-bar">
          <div class="score-fill ${getScoreClass(misalignmentPercent)}" style="width: ${misalignmentPercent}%"></div>
        </div>
      </div>
      
      <div class="promptguard-section">
        <h3>Original Prompt</h3>
        <div class="prompt-box original">${escapeHtml(original)}</div>
      </div>
      
      <div class="promptguard-section">
        <h3>Enhanced Prompt</h3>
        <div class="prompt-box enhanced">${escapeHtml(enhanced)}</div>
      </div>
      
      <div class="promptguard-actions">
        <button class="btn-secondary promptguard-close-btn">Keep Original</button>
        <button class="btn-primary promptguard-use-btn">Use Enhanced</button>
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  
  modal.querySelector('.promptguard-close').addEventListener('click', () => modal.remove());
  modal.querySelector('.promptguard-close-btn').addEventListener('click', () => modal.remove());
  modal.querySelector('.promptguard-use-btn').addEventListener('click', () => {
    replacePrompt(inputElement, enhanced);
    modal.remove();
    showNotification('Enhanced prompt applied!', 'success');
  });
  
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.remove();
  });
}

function replacePrompt(element, newText) {
  if (!element) return;
  
  if (element.tagName === 'TEXTAREA') {
    element.value = newText;
    element.dispatchEvent(new Event('input', { bubbles: true }));
  } else {
    element.innerText = newText;
    element.dispatchEvent(new Event('input', { bubbles: true }));
  }
  
  element.focus();
}

function showNotification(message, type = 'info') {
  const existing = document.getElementById('promptguard-notification');
  if (existing) existing.remove();
  
  const notification = document.createElement('div');
  notification.id = 'promptguard-notification';
  notification.className = `promptguard-notification ${type}`;
  notification.textContent = message;
  
  document.body.appendChild(notification);
  
  setTimeout(() => notification.classList.add('show'), 10);
  
  if (type !== 'loading') {
    setTimeout(() => {
      notification.classList.remove('show');
      setTimeout(() => notification.remove(), 300);
    }, 3000);
  }
}

function getScoreClass(percent) {
  if (percent < 20) return 'score-low';
  if (percent < 50) return 'score-medium';
  return 'score-high';
}


function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function init() {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
    return;
  }
  
  setTimeout(createPromptGuardButton, 1000);
  
  setInterval(() => {
    if (!document.getElementById('promptguard-button')) {
      createPromptGuardButton();
    }
  }, 3000);
}

init();
// showResults();

// async function listAvailableModels(apiKey) {
//   try {
//     console.log('Testing v1beta...');
//     let response = await fetch(
//       `https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`
//     );
    
//     let data = await response.json();
//     console.log('v1beta Response:', data);
    
//     if (!response.ok) {
//       console.log('v1beta failed, trying v1...');
//       response = await fetch(
//         `https://generativelanguage.googleapis.com/v1/models?key=${apiKey}`
//       );
//       data = await response.json();
//       console.log('v1 Response:', data);
//     }
    
//     // Check different possible response structures
//     if (data.models) {
//       console.log('✅ Available models:', data.models.map(m => m.name));
//     } else if (data.model) {
//       console.log('✅ Single model:', data.model);
//     } else {
//       console.log('❌ Full response:', JSON.stringify(data, null, 2));
//     }
    
//     return data;
//   } catch (error) {
//     console.error('❌ Error:', error);
//   }
// }

// listAvailableModels(apiKey);