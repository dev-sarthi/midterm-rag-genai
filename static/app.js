/**
 * Designs Law Assistant - Frontend Logic
 * Handles interactive queries, suggested pills, keyboard events, and citation rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  const askForm = document.getElementById('askForm');
  const questionInput = document.getElementById('questionInput');
  const submitBtn = document.getElementById('submitBtn');
  const clearBtn = document.getElementById('clearBtn');
  const suggestedChips = document.getElementById('suggestedChips');

  const errorAlert = document.getElementById('errorAlert');
  const errorTitle = document.getElementById('errorTitle');
  const errorMessage = document.getElementById('errorMessage');
  const dismissErrorBtn = document.getElementById('dismissErrorBtn');

  const loadingState = document.getElementById('loadingState');
  const resultsSection = document.getElementById('resultsSection');
  const questionEcho = document.getElementById('questionEcho');
  const answerBody = document.getElementById('answerBody');
  const notFoundBox = document.getElementById('notFoundBox');
  const sourcesGrid = document.getElementById('sourcesGrid');
  const sourceCount = document.getElementById('sourceCount');
  const copyBtn = document.getElementById('copyBtn');
  const copyBtnText = document.getElementById('copyBtnText');

  let currentRawAnswer = '';

  // 1. Keyboard shortcuts: Enter submits, Shift+Enter inserts newline
  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askForm.requestSubmit();
    }
  });

  // 2. Clear input
  clearBtn.addEventListener('click', () => {
    questionInput.value = '';
    questionInput.focus();
  });

  // 3. Dismiss error alert
  dismissErrorBtn.addEventListener('click', () => {
    errorAlert.classList.add('hidden');
  });

  // 4. Clickable Suggested Questions
  suggestedChips.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    const questionText = chip.getAttribute('data-question');
    if (questionText) {
      questionInput.value = questionText;
      askForm.requestSubmit();
    }
  });

  // 5. Copy answer text
  copyBtn.addEventListener('click', async () => {
    if (!currentRawAnswer) return;
    try {
      await navigator.clipboard.writeText(currentRawAnswer);
      const originalText = copyBtnText.textContent;
      copyBtnText.textContent = 'Copied!';
      copyBtn.classList.add('btn-primary');
      setTimeout(() => {
        copyBtnText.textContent = originalText;
        copyBtn.classList.remove('btn-primary');
      }, 2000);
    } catch (err) {
      console.warn('Clipboard write failed:', err);
    }
  });

  // 6. Form Submission
  askForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = questionInput.value.trim();

    if (!query) {
      showError('Input Required', 'Please enter a valid legal question.');
      questionInput.focus();
      return;
    }

    // Set UI to loading state and disable duplicate submissions
    setLoading(true);
    hideError();

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: query }),
      });

      if (!response.ok) {
        let errDetail = 'Failed to fetch statutory answer.';
        try {
          const errData = await response.json();
          if (errData.detail) errDetail = errData.detail;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      console.error('Error during query:', err);
      showError('Error Generating Answer', err.message || 'Network error or service unavailable.');
      resultsSection.classList.add('hidden');
    } finally {
      setLoading(false);
    }
  });

  // UI State Helpers
  function setLoading(isLoading) {
    if (isLoading) {
      submitBtn.disabled = true;
      submitBtn.classList.add('loading');
      questionInput.disabled = true;
      loadingState.classList.remove('hidden');
      resultsSection.classList.add('hidden');
    } else {
      submitBtn.disabled = false;
      submitBtn.classList.remove('loading');
      questionInput.disabled = false;
      loadingState.classList.add('hidden');
    }
  }

  function showError(title, msg) {
    errorTitle.textContent = title;
    errorMessage.textContent = msg;
    errorAlert.classList.remove('hidden');
  }

  function hideError() {
    errorAlert.classList.add('hidden');
  }

  // Formatting markdown text into clean HTML safely
  function formatMarkdown(text) {
    if (!text) return '';

    // Escape HTML to prevent injection
    let safe = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Bold formatting: **text**
    safe = safe.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Bullet points: lines starting with * or -
    const lines = safe.split('\n');
    let inList = false;
    let html = '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
        if (!inList) {
          html += '<ul>';
          inList = true;
        }
        html += `<li>${trimmed.substring(2)}</li>`;
      } else if (/^\d+\.\s/.test(trimmed)) {
        // Numbered list items
        if (!inList) {
          html += '<ol>';
          inList = true;
        }
        html += `<li>${trimmed.replace(/^\d+\.\s/, '')}</li>`;
      } else {
        if (inList) {
          html += '</ul>';
          inList = false;
        }
        if (trimmed.length > 0) {
          html += `<p>${trimmed}</p>`;
        }
      }
    }

    if (inList) {
      html += '</ul>';
    }

    return html;
  }

  function renderResults(data) {
    const { question, answer, sources } = data;
    currentRawAnswer = answer;

    questionEcho.textContent = `Query: "${question}"`;

    const isNotFound = answer.trim().toLowerCase() === 'not found';

    if (isNotFound) {
      notFoundBox.classList.remove('hidden');
      answerBody.classList.add('hidden');
      copyBtn.classList.add('hidden');
    } else {
      notFoundBox.classList.add('hidden');
      answerBody.classList.remove('hidden');
      copyBtn.classList.remove('hidden');
      answerBody.innerHTML = formatMarkdown(answer);
    }

    // Render Source Cards
    sourcesGrid.innerHTML = '';
    const validSources = sources || [];
    sourceCount.textContent = validSources.length;

    validSources.forEach((src) => {
      const card = document.createElement('div');
      card.className = 'source-card';

      const pageText = `Page ${src.page}`;
      const sectionText = src.section || 'General';

      card.innerHTML = `
        <div class="source-card-header">
          <div class="source-meta-tags">
            <span class="meta-chip meta-chip-page">${escapeHtml(pageText)}</span>
            <span class="meta-chip meta-chip-section">${escapeHtml(sectionText)}</span>
          </div>
          <span class="source-index">Chunk #${src.chunk_index}</span>
        </div>
        <div class="source-excerpt" title="Retrieved passage text">${escapeHtml(src.content)}</div>
      `;
      sourcesGrid.appendChild(card);
    });

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
