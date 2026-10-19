/* FakeGuard AI — Frontend JavaScript */

// ─── Sample Data ─────────────────────────────────────────────────────────────
const SAMPLES = {
  real: {
    title: "Federal Reserve raises interest rates by 25 basis points amid inflation concerns",
    text: "The Federal Reserve's monetary policy committee voted unanimously on Wednesday to raise the benchmark interest rate by 25 basis points to 5.5%, the highest level in 22 years. Chair Jerome Powell stated at the post-meeting press conference that officials remain committed to bringing inflation down to the 2% target. The decision came as consumer price data showed inflation easing to 3.1% in recent months, though still above the central bank's goal. Markets had widely anticipated the move. Analysts expect the Fed to hold rates steady at upcoming meetings while assessing the impact on economic growth and employment."
  },
  fake: {
    title: "BOMBSHELL: Government SECRETLY inserting 5G microchips into flu vaccines – LEAKED documents PROVE it!!!",
    text: "URGENT: A whistleblower from inside the deep state has leaked classified documents PROVING that the government has been secretly inserting 5G microchips into flu vaccines since 2019! The anonymous source, whose identity cannot be verified for safety reasons, claims that these nanobots are activated by 5G towers to control your thoughts and monitor your location. BIG PHARMA doesn't want you to know this! SHARE THIS BEFORE IT GETS DELETED! The mainstream media is SUPPRESSING this story because they are paid by the elites. Wake up sheeple! Do your own research! Trust me bro, this is 100% real. If this gets 50,000 shares maybe the truth will finally come out!"
  }
};

// ─── DOM References ───────────────────────────────────────────────────────────
const titleInput   = document.getElementById('newsTitle');
const textInput    = document.getElementById('newsText');
const charCount    = document.getElementById('charCount');
const analyzeBtn   = document.getElementById('analyzeBtn');
const analyzeBtnTxt = document.getElementById('analyzeBtnText');
const clearBtn     = document.getElementById('clearBtn');

const idleState    = document.getElementById('idleState');
const loadingState = document.getElementById('loadingState');
const resultState  = document.getElementById('resultState');

// ─── Helpers ──────────────────────────────────────────────────────────────────
function showState(state) {
  idleState.classList.add('hidden');
  loadingState.classList.add('hidden');
  resultState.classList.add('hidden');
  state.classList.remove('hidden');
}

function updateCharCount() {
  const total = (titleInput.value + textInput.value).length;
  charCount.textContent = total.toLocaleString();
}

function loadSample(type) {
  const sample = SAMPLES[type];
  titleInput.value = sample.title;
  textInput.value  = sample.text;
  updateCharCount();
  titleInput.dispatchEvent(new Event('input'));
  // Auto-scroll to button
  analyzeBtn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setLoading(loading) {
  analyzeBtn.disabled = loading;
  if (loading) {
    analyzeBtnTxt.textContent = 'Analyzing...';
    showState(loadingState);
  } else {
    analyzeBtnTxt.textContent = 'Analyze Article';
  }
}

// ─── Confidence Ring Animation ────────────────────────────────────────────────
function animateConfRing(confidence, isReal) {
  const ring = document.getElementById('confRingProgress');
  const confVal = document.getElementById('confValue');
  const circumference = 2 * Math.PI * 15.9; // r=15.9

  // Reset
  ring.style.strokeDasharray = '0 100';
  confVal.textContent = '0%';

  setTimeout(() => {
    const progress = (confidence / 100) * 100;
    ring.style.strokeDasharray = `${progress} 100`;

    // Count up animation
    let current = 0;
    const step = confidence / 40;
    const timer = setInterval(() => {
      current = Math.min(current + step, confidence);
      confVal.textContent = Math.round(current) + '%';
      if (current >= confidence) clearInterval(timer);
    }, 25);
  }, 100);
}

// ─── Render Results ───────────────────────────────────────────────────────────
function renderResults(data) {
  const isReal = data.label === 'REAL';

  // Verdict Banner
  const banner = document.getElementById('verdictBanner');
  banner.className = 'verdict-banner ' + (isReal ? 'verdict-real' : 'verdict-fake');

  document.getElementById('verdictIcon').innerHTML = isReal
    ? '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>'
    : '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';

  document.getElementById('verdictLabel').textContent = 'AI VERDICT';
  document.getElementById('verdictValue').textContent = isReal ? '✓ REAL NEWS' : '✗ FAKE NEWS';

  // Confidence ring
  animateConfRing(data.confidence, isReal);

  // Text stats
  const stats = data.text_stats;
  document.getElementById('statWords').textContent    = stats.word_count || '—';
  document.getElementById('statExclaim').textContent  = stats.exclamation_marks ?? '—';
  document.getElementById('statCaps').textContent     = (stats.uppercase_ratio ?? '—') + '%';
  document.getElementById('statModel').textContent    = (data.model_used || '—').split(' ').map(w => w[0]).join('');

  // Red Flags
  const redFlagsSection = document.getElementById('redFlagsSection');
  const redFlagsList    = document.getElementById('redFlagsList');
  redFlagsList.innerHTML = '';

  if (data.red_flags && data.red_flags.length > 0) {
    redFlagsSection.classList.remove('hidden');
    data.red_flags.forEach(flag => {
      const item = document.createElement('div');
      item.className = `red-flag-item red-flag-${flag.severity}`;
      item.innerHTML = `
        <div>
          <div class="red-flag-name">${escHtml(flag.flag)}</div>
          <div class="red-flag-detail">${escHtml(flag.detail)}</div>
        </div>`;
      redFlagsList.appendChild(item);
    });
  } else {
    redFlagsSection.classList.add('hidden');
  }

  // Keywords
  const keyList = document.getElementById('keywordsList');
  keyList.innerHTML = '';
  if (data.top_words && data.top_words.length > 0) {
    data.top_words.forEach(kw => {
      const tag = document.createElement('span');
      tag.className = 'keyword-tag';
      tag.textContent = kw.word;
      tag.title = `TF-IDF score: ${kw.score}`;
      keyList.appendChild(tag);
    });
  } else {
    keyList.innerHTML = '<span style="color:var(--text-muted);font-size:0.82rem">No significant keywords found</span>';
  }

  showState(resultState);
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ─── Analyze ──────────────────────────────────────────────────────────────────
async function analyzeArticle() {
  const title = titleInput.value.trim();
  const text  = textInput.value.trim();

  if (!title && !text) {
    titleInput.focus();
    titleInput.style.borderColor = 'var(--red)';
    titleInput.style.boxShadow = '0 0 0 3px rgba(239,68,68,0.15)';
    setTimeout(() => {
      titleInput.style.borderColor = '';
      titleInput.style.boxShadow = '';
    }, 2000);
    return;
  }

  setLoading(true);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, text })
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || 'Server error');
    }

    renderResults(data);

  } catch (err) {
    showState(idleState);
    showErrorBanner(err.message || 'Failed to analyze. Make sure the server is running.');
  } finally {
    setLoading(false);
  }
}

function showErrorBanner(msg) {
  // Remove old banner if any
  document.querySelectorAll('.error-toast').forEach(e => e.remove());
  const toast = document.createElement('div');
  toast.className = 'error-toast';
  toast.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:9999;
    background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.3);
    color:#f87171; padding:14px 20px; border-radius:10px;
    font-size:0.875rem; max-width:360px;
    animation: fadeInUp 0.3s ease;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  `;
  toast.textContent = '⚠ ' + msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}

// ─── Event Listeners ──────────────────────────────────────────────────────────
titleInput.addEventListener('input', updateCharCount);
textInput.addEventListener('input', updateCharCount);

clearBtn.addEventListener('click', () => {
  titleInput.value = '';
  textInput.value  = '';
  updateCharCount();
  showState(idleState);
});

analyzeBtn.addEventListener('click', analyzeArticle);

// Enter key in title field
titleInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    analyzeArticle();
  }
});

// Smooth scroll for nav links
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', (e) => {
    const target = document.querySelector(link.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Animate elements on scroll (IntersectionObserver)
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.model-card, .chart-card, .pipeline-step').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(24px)';
  el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  observer.observe(el);
});

// Init
updateCharCount();
