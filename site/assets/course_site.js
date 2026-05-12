const DB = window.COURSE_DB;
const state = { query: "", selectedTerm: "" };

const $ = (id) => document.getElementById(id);
const norm = (s) => (s || "").toString().toLowerCase();
const escapeHtml = (s) => (s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function highlight(text, query) {
  const safe = escapeHtml(text || "");
  const q = (query || "").trim();
  if (!q) return safe;
  const parts = q.split(/\s+/).filter(Boolean).slice(0, 6).map(x => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!parts.length) return safe;
  return safe.replace(new RegExp(`(${parts.join("|")})`, "gi"), "<mark>$1</mark>");
}

function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("active", b.dataset.view === name));
}

function scoreChunk(chunk, query) {
  const q = norm(query).split(/\s+/).filter(Boolean);
  if (!q.length) return 0;
  const title = norm(chunk.title + " " + chunk.heading_path.join(" "));
  const text = norm(chunk.text);
  const terms = norm((chunk.terms || []).join(" "));
  let score = 0;
  q.forEach(part => {
    if (title.includes(part)) score += 10;
    if (terms.includes(part)) score += 8;
    const matches = text.match(new RegExp(part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"));
    if (matches) score += Math.min(matches.length, 12);
  });
  return score;
}

function renderStats() {
  const distinctRandomPapers = Math.min(
    Math.floor(DB.quizBank.filter(q => q.type === "mcq").length / 20),
    Math.floor(DB.quizBank.filter(q => q.type === "fill").length / 10),
    Math.floor(DB.quizBank.filter(q => q.type === "short").length / 5),
  );
  $("stats").innerHTML = `
    <div class="stat"><strong>${DB.manifest.length}</strong><span>课件文件</span></div>
    <div class="stat"><strong>${DB.chunks.length}</strong><span>可搜索片段</span></div>
    <div class="stat"><strong>${Object.keys(DB.terms).length}</strong><span>索引词条</span></div>
    <div class="stat"><strong>${DB.quizBank.length}</strong><span>结构化题目</span></div>
    <div class="stat"><strong>${distinctRandomPapers}</strong><span>可支撑不重复整卷数</span></div>
  `;
}

function renderTermCloud() {
  const terms = Object.values(DB.terms).sort((a, b) => b.total_count - a.total_count).slice(0, 40);
  $("termCloud").innerHTML = terms.map(t => `
    <button class="term-btn" data-term="${escapeHtml(t.term)}">
      <strong>${escapeHtml(t.term)}</strong>
      <span>${t.total_count} 次 · ${t.files.length} 文件</span>
    </button>
  `).join("");
  $("termCloud").querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    $("searchInput").value = btn.dataset.term;
    runSearch(btn.dataset.term);
  }));
}

function explainQuery(query, results) {
  const exact = DB.terms[query];
  const fuzzy = exact || Object.values(DB.terms).find(t => norm(t.term).includes(norm(query)) || norm(query).includes(norm(t.term)));
  if (!query.trim()) {
    $("explainBox").innerHTML = `<h2>搜索说明</h2><p>输入词条后，会显示它在所有课件中的出现位置、所属章节和相关知识点。点击词条或结果可以生成相关复习题。</p>`;
    return;
  }
  if (fuzzy) {
    $("explainBox").innerHTML = `
      <h2>${escapeHtml(fuzzy.term)}</h2>
      <p>该词条在 <strong>${fuzzy.files.length}</strong> 个课件文件中出现，共索引到 <strong>${fuzzy.total_count}</strong> 次。下面列出最相关的位置和相邻知识点。</p>
      <div class="chips">${(fuzzy.related || []).map(r => `<button class="chip" data-term="${escapeHtml(r)}">${escapeHtml(r)}</button>`).join("")}</div>
      <button class="btn secondary" id="quizThisTerm">生成这个词条的复习题</button>
    `;
    $("explainBox").querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", () => {
      $("searchInput").value = chip.dataset.term;
      runSearch(chip.dataset.term);
    }));
    $("quizThisTerm").addEventListener("click", () => generateQuiz(fuzzy.term));
  } else {
    $("explainBox").innerHTML = `<h2>${escapeHtml(query)}</h2><p>没有找到精确词条索引，但全文搜索得到 <strong>${results.length}</strong> 个相关片段。可以根据这些片段生成综合复习题。</p><button class="btn secondary" id="quizThisTerm">生成相关复习题</button>`;
    $("quizThisTerm").addEventListener("click", () => generateQuiz(query));
  }
}

function runSearch(query) {
  state.query = query;
  const scored = DB.chunks.map(c => [scoreChunk(c, query), c]).filter(([s]) => s > 0).sort((a, b) => b[0] - a[0]).slice(0, 80);
  const results = scored.map(([, c]) => c);
  explainQuery(query, results);
  $("resultCount").textContent = query.trim() ? `${results.length} 个结果` : "等待输入";
  $("results").innerHTML = results.length ? results.map(c => `
    <article class="result">
      <div class="result-title">
        <span>${escapeHtml(c.title || c.file)}</span>
        <button class="chip result-quiz" data-topic="${escapeHtml((c.terms && c.terms[0]) || query)}">出题</button>
      </div>
      <div class="path">${escapeHtml((c.heading_path || []).join(" / "))}</div>
      <p class="snippet">${highlight(c.summary, query)}</p>
      <span class="source">${escapeHtml(c.file)} · cell ${c.cell_index} · ${escapeHtml(c.type)}</span>
      <div class="chips">${(c.terms || []).slice(0, 8).map(t => `<button class="chip" data-term="${escapeHtml(t)}">${escapeHtml(t)}</button>`).join("")}</div>
    </article>
  `).join("") : `<div class="panel"><p class="meta">输入课件中的词条，例如“感受野”“BatchNorm”“LSTM”“CrossEntropyLoss”。</p></div>`;
  $("results").querySelectorAll(".chip[data-term]").forEach(chip => chip.addEventListener("click", () => {
    $("searchInput").value = chip.dataset.term;
    runSearch(chip.dataset.term);
  }));
  $("results").querySelectorAll(".result-quiz").forEach(btn => btn.addEventListener("click", () => generateQuiz(btn.dataset.topic)));
}

function pickQuestions(type, topic, count) {
  const q = norm(topic || "");
  const all = DB.quizBank.filter(x => x.type === type);
  const related = all.filter(x => norm([x.topic, x.stem, x.source].join(" ")).includes(q));
  const fallback = all.filter(x => !related.includes(x));
  const shuffle = (arr) => {
    const copy = [...arr];
    for (let i = copy.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  };
  let selected = [];
  if (q) {
    const desiredRelated = Math.min(count, Math.max(Math.ceil(count * 0.7), Math.min(related.length, 3)));
    selected = shuffle(related).slice(0, desiredRelated);
    if (selected.length < count) {
      selected = selected.concat(shuffle(fallback).slice(0, count - selected.length));
    }
  } else {
    const hardQuota = type === "mcq" ? Math.min(3, count) : 0;
    const hard = shuffle(all.filter(x => x.difficulty === "难")).slice(0, hardQuota);
    const rest = shuffle(all.filter(x => !hard.includes(x))).slice(0, count - hard.length);
    selected = hard.concat(rest);
  }
  return selected.slice(0, count);
}

function generateQuiz(topic = "") {
  const mcq = pickQuestions("mcq", topic, 20);
  const fill = pickQuestions("fill", topic, 10);
  const short = pickQuestions("short", topic, 5);
  const title = topic ? `复习题：${topic}` : "综合随机复习题";
  const chosen = [...mcq, ...fill, ...short];
  $("quizTitle").textContent = title;
  $("quizMeta").textContent = `本次随机抽取 ${mcq.length} 道单选、${fill.length} 道填空、${short.length} 道简答。再次点击会重新随机生成。`;
  $("quizBody").innerHTML = `
    <section class="quiz-section"><h3>一、单选题</h3>${mcq.map((q, i) => renderQuestion(q, i + 1)).join("")}</section>
    <section class="quiz-section"><h3>二、填空题</h3>${fill.map((q, i) => renderQuestion(q, i + 1)).join("")}</section>
    <section class="quiz-section"><h3>三、简答题</h3>${short.map((q, i) => renderQuestion(q, i + 1)).join("")}</section>
    <section class="panel answers"><h3>答案与解析</h3>${chosen.map((q, i) => renderAnswer(q, i + 1)).join("")}</section>
  `;
  showView("quiz");
  window.scrollTo({top: 0, behavior: "smooth"});
}

function renderQuestion(q, idx) {
  if (q.type === "mcq") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><ul class="options">${q.options.map(o => `<li>${escapeHtml(o)}</li>`).join("")}</ul></div>`;
  }
  if (q.type === "fill") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="blank-answer"></span> <span class="badge">${escapeHtml(q.difficulty)}</span></div></div>`;
  }
  return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><p class="meta">答题区：</p><p style="height:54px;border-bottom:1px solid var(--line)"></p></div>`;
}

function renderAnswer(q, idx) {
  return `<div class="answer-item"><strong>${idx}. [${escapeHtml(q.id)}] ${escapeHtml(q.answer)}</strong><p>${escapeHtml(q.explanation)}</p><span class="source">来源：${escapeHtml(q.source)} · ${escapeHtml(q.topic)}</span></div>`;
}

function renderReview() {
  const r = DB.review;
  $("reviewBody").innerHTML = r.sections.map(sec => `
    <section class="review-section">
      <h2>${escapeHtml(sec.title)}</h2>
      <ul>${sec.points.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>
      ${sec.details ? `<h3>详细知识点</h3><ul>${sec.details.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${sec.must_memorize ? `<h3>必背结论</h3><ul>${sec.must_memorize.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${(sec.formulas || []).map(f => `<div class="formula">${escapeHtml(f)}</div>`).join("")}
      ${sec.formula_explainer ? `<h3>公式拆解</h3><ul>${sec.formula_explainer.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      <p><strong>常考点：</strong>${escapeHtml((sec.exam_focus || []).join("；"))}</p>
      ${sec.question_patterns ? `<p><strong>命题方式：</strong>${escapeHtml(sec.question_patterns.join("；"))}</p>` : ""}
      ${sec.answer_template ? `<p><strong>答题模板：</strong>${escapeHtml(sec.answer_template.join("；"))}</p>` : ""}
      ${sec.traps ? `<p><strong>易错陷阱：</strong>${escapeHtml(sec.traps.join("；"))}</p>` : ""}
      ${sec.images ? `<div class="review-images">${sec.images.map(src => `<img src="${escapeHtml(src)}" alt="${escapeHtml(sec.title)}">`).join("")}</div>` : ""}
    </section>
  `).join("") + (r.references ? `<section class="review-section"><h2>参考资料</h2><ul>${r.references.map(ref => `<li><a href="${escapeHtml(ref.url)}">${escapeHtml(ref.title)}</a></li>`).join("")}</ul></section>` : "");
}

function init() {
  document.querySelectorAll("nav button").forEach(btn => btn.addEventListener("click", () => showView(btn.dataset.view)));
  $("searchInput").addEventListener("input", (e) => runSearch(e.target.value));
  $("makeQuiz").addEventListener("click", () => generateQuiz($("searchInput").value.trim()));
  $("makeFullQuiz").addEventListener("click", () => generateQuiz(""));
  $("quizAll").addEventListener("click", () => generateQuiz(""));
  renderStats();
  renderTermCloud();
  renderReview();
  runSearch("");
}

init();
