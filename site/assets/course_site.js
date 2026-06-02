const DB = window.COURSE_DB;
const $ = (id) => document.getElementById(id);
const norm = (s) => (s || "").toString().toLowerCase();
const escapeHtml = (s) => (s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("active", b.dataset.view === name));
}

function highlight(text, query) {
  const safe = escapeHtml(text || "");
  const q = (query || "").trim();
  if (!q) return safe;
  const parts = q.split(/\s+/).filter(Boolean).slice(0, 6).map(x => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!parts.length) return safe;
  return safe.replace(new RegExp(`(${parts.join("|")})`, "gi"), "<mark>$1</mark>");
}

function scoreChunk(chunk, query) {
  const q = norm(query).split(/\s+/).filter(Boolean);
  if (!q.length) return 0;
  const title = norm(chunk.title + " " + (chunk.heading_path || []).join(" "));
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

function shuffle(arr) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function renderStats() {
  $("stats").innerHTML = `
    <div class="stat"><strong>${DB.manifest.length}</strong><span>课件文件</span></div>
    <div class="stat"><strong>${DB.chunks.length}</strong><span>可检索片段</span></div>
    <div class="stat"><strong>${Object.keys(DB.terms).length}</strong><span>索引词条</span></div>
    <div class="stat"><strong>${DB.quizBank.length}</strong><span>标准题</span></div>
    <div class="stat"><strong>${DB.beginnerQuizBank.length}</strong><span>基础题</span></div>
  `;
}

function renderTermCloud() {
  const terms = Object.values(DB.terms).sort((a, b) => b.total_count - a.total_count).slice(0, 42);
  $("termCloud").innerHTML = terms.map(t => `
    <button class="term-btn" data-term="${escapeHtml(t.term)}">
      <strong>${escapeHtml(t.term)}</strong>
      <span>${t.total_count} 次 · ${t.files.length} 文件</span>
    </button>
  `).join("");
  $("termCloud").querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    $("searchInput").value = btn.dataset.term;
    runSearch(btn.dataset.term);
    showView("search");
  }));
}

function runSearch(query) {
  const scored = DB.chunks.map(c => [scoreChunk(c, query), c]).filter(([s]) => s > 0).sort((a, b) => b[0] - a[0]).slice(0, 80);
  const results = scored.map(([, c]) => c);
  $("resultCount").textContent = query.trim() ? `${results.length} 个结果` : "等待输入";
  $("results").innerHTML = results.length ? results.map(c => `
    <article class="result">
      <div class="result-title">
        <span>${escapeHtml(c.title || c.file)}</span>
        <button class="chip result-quiz" data-topic="${escapeHtml((c.terms && c.terms[0]) || query)}">出题</button>
      </div>
      <div class="path">${escapeHtml((c.heading_path || []).join(" / "))}</div>
      <p class="snippet">${highlight(c.summary, query)}</p>
      <span class="meta">${escapeHtml(c.file)} · ${c.slide_index ? `第 ${c.slide_index} 页` : `cell ${c.cell_index}`} · ${escapeHtml(c.kind || c.type)}</span>
      <div class="chips">${(c.terms || []).slice(0, 8).map(t => `<button class="chip" data-term="${escapeHtml(t)}">${escapeHtml(t)}</button>`).join("")}</div>
    </article>
  `).join("") : `<div class="panel"><p class="meta">输入课件中的词条，例如“感受野”“BatchNorm”“LSTM”“CrossEntropyLoss”。</p></div>`;
  $("results").querySelectorAll(".chip[data-term]").forEach(chip => chip.addEventListener("click", () => {
    $("searchInput").value = chip.dataset.term;
    runSearch(chip.dataset.term);
  }));
  $("results").querySelectorAll(".result-quiz").forEach(btn => btn.addEventListener("click", () => generateStandardQuiz(btn.dataset.topic)));

  const fuzzy = DB.terms[query] || Object.values(DB.terms).find(t => query && (norm(t.term).includes(norm(query)) || norm(query).includes(norm(t.term))));
  $("explainBox").innerHTML = fuzzy ? `
    <h2>${escapeHtml(fuzzy.term)}</h2>
    <p class="muted">出现 ${fuzzy.total_count} 次，分布在 ${fuzzy.files.length} 个课件文件中。</p>
    <div class="chips">${(fuzzy.related || []).map(r => `<button class="chip" data-term="${escapeHtml(r)}">${escapeHtml(r)}</button>`).join("")}</div>
  ` : `<h2>搜索说明</h2><p class="muted">搜索会定位到课件原文片段、相关词条和来源位置。适合先查概念，再回到教材目录和导图复习。</p>`;
  $("explainBox").querySelectorAll(".chip[data-term]").forEach(chip => chip.addEventListener("click", () => {
    $("searchInput").value = chip.dataset.term;
    runSearch(chip.dataset.term);
  }));
}

function renderOutline() {
  const courses = DB.outline.courses;
  $("courseIndex").innerHTML = courses.map((c, i) => `<a href="#course-${i}">${escapeHtml(c.title)}</a>`).join("");
  $("outlineBody").innerHTML = courses.map((course, i) => `
    <section class="panel course-card" id="course-${i}">
      <div class="course-head">
        <div>
          <h2>${escapeHtml(course.title)}</h2>
          <p class="muted">${escapeHtml(course.file)} · ${course.chapter_count} 个章节块 · ${course.chunk_count} 个片段</p>
        </div>
        <button class="btn secondary" data-map="${escapeHtml(course.file)}">看导图</button>
      </div>
      ${course.chapters.map(ch => `
        <div class="chapter">
          <h3>${escapeHtml(ch.title)}</h3>
          <table>
            <thead><tr><th>知识点</th><th>通俗解释</th><th>考试怎么考</th><th>易错点</th><th>必记句子/公式</th><th>来源</th></tr></thead>
            <tbody>${ch.rows.map(r => `
              <tr>
                <td><strong>${escapeHtml(r.term)}</strong></td>
                <td>${escapeHtml(r.plain)}</td>
                <td>${escapeHtml(r.exam)}</td>
                <td>${escapeHtml(r.pitfall)}</td>
                <td>${escapeHtml(r.memory)}</td>
                <td class="muted">${escapeHtml(r.source)}</td>
              </tr>
            `).join("")}</tbody>
          </table>
        </div>
      `).join("")}
    </section>
  `).join("");
  $("outlineBody").querySelectorAll("button[data-map]").forEach(btn => btn.addEventListener("click", () => {
    showView("mindmaps");
    const card = document.querySelector(`[data-map-card="${CSS.escape(btn.dataset.map)}"]`);
    if (card) card.scrollIntoView({behavior: "smooth", block: "start"});
  }));
}

function renderMapTree(map) {
  return `<div class="map-tree"><ul>${map.chapters.map(ch => `
    <li><span class="map-node">${escapeHtml(ch.title)}</span>
      <ul>${(ch.terms || []).slice(0, 16).map(t => `<li><span class="map-node term" data-term="${escapeHtml(t)}">${escapeHtml(t)}</span></li>`).join("")}</ul>
    </li>
  `).join("")}</ul></div>`;
}

function renderMindmaps() {
  $("mindmapBody").innerHTML = DB.mindmaps.map(course => `
    <article class="panel map-card" data-map-card="${escapeHtml(course.file)}">
      <h3>${escapeHtml(course.title)}</h3>
      <p class="muted">${escapeHtml(course.file)} · ${course.maps.length} 张蜘蛛网导图${course.maps.length > 1 ? "，已按章节拆分" : ""}</p>
      ${course.maps.slice(0, 1).map(map => `
        <div class="chapter">
          <div class="result-title">
            <strong>${escapeHtml(map.title)} <span class="badge">${map.node_count} 节点</span></strong>
            <span class="map-actions">
              <a class="btn secondary" href="${escapeHtml(map.svg)}" target="_blank">打开大图</a>
              <a class="btn secondary" href="${escapeHtml(map.svg)}" download>下载 SVG</a>
              <a class="btn secondary" href="${escapeHtml(map.markdown)}" download>下载大纲</a>
            </span>
          </div>
          <div class="map-preview"><img src="${escapeHtml(map.svg)}" alt="${escapeHtml(map.title)} 思维导图"></div>
          <div class="map-tree compact-tree">${renderMapTree(map)}</div>
        </div>
      `).join("")}
      ${course.maps.length > 1 ? `
        <details class="submaps">
          <summary>展开章节小图和大纲下载（${course.maps.length - 1} 张）</summary>
          ${course.maps.slice(1).map(map => `
            <div class="submap-row">
              <strong>${escapeHtml(map.title)}</strong>
              <span class="map-actions">
                <a href="${escapeHtml(map.svg)}" target="_blank">打开</a>
                <a href="${escapeHtml(map.svg)}" download>下载 SVG</a>
                <a href="${escapeHtml(map.markdown)}" download>下载大纲</a>
              </span>
            </div>
          `).join("")}
        </details>` : ""}
    </article>
  `).join("");
  $("mindmapBody").querySelectorAll(".map-node.term").forEach(node => node.addEventListener("click", () => {
    $("searchInput").value = node.dataset.term;
    runSearch(node.dataset.term);
    showView("search");
  }));
}

function pickQuestions(bank, type, topic, count) {
  const q = norm(topic || "");
  const all = bank.filter(x => x.type === type);
  const related = q ? all.filter(x => norm([x.topic, x.stem, x.source].join(" ")).includes(q)) : [];
  const base = related.length >= Math.min(count, 3) ? related : all;
  return shuffle(base).slice(0, count);
}

function generateQuiz(bank, mode, topic = "") {
  const spec = mode === "beginner"
    ? [["mcq", 10], ["tf", 10], ["fill", 10], ["short", 3]]
    : [["mcq", 20], ["fill", 10], ["short", 5]];
  const labels = {mcq: "单选题", tf: "判断题", fill: "填空题", short: "简答题"};
  const chosen = spec.flatMap(([type, count]) => pickQuestions(bank, type, topic, count));
  const title = mode === "beginner" ? "基础练习随机题" : "标准综合随机题";
  const targetTitle = mode === "beginner" ? $("beginnerTitle") : $("quizTitle");
  const targetMeta = mode === "beginner" ? $("beginnerMeta") : $("quizMeta");
  const targetBody = mode === "beginner" ? $("beginnerBody") : $("quizBody");
  targetTitle.textContent = topic ? `${title}：${topic}` : title;
  targetMeta.textContent = spec.map(([type, count]) => `${count} 道${labels[type]}`).join("、") + "。再次点击会重新随机生成。";
  targetBody.innerHTML = spec.map(([type]) => {
    const items = chosen.filter(q => q.type === type);
    return `<section class="quiz-section"><h3>${labels[type]}</h3>${items.map((q, i) => renderQuestion(q, i + 1)).join("")}</section>`;
  }).join("") + `<section class="panel answers"><h3>答案与解析</h3>${chosen.map((q, i) => renderAnswer(q, i + 1)).join("")}</section>`;
}

function generateStandardQuiz(topic = "", activate = true) {
  generateQuiz(DB.quizBank, "standard", topic);
  if (activate) {
    showView("quiz");
    window.scrollTo({top: 0, behavior: "smooth"});
  }
}

function generateBeginnerQuiz(topic = "", activate = true) {
  generateQuiz(DB.beginnerQuizBank, "beginner", topic);
  if (activate) {
    showView("beginner");
    window.scrollTo({top: 0, behavior: "smooth"});
  }
}

function renderQuestion(q, idx) {
  if (q.type === "mcq") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><ul class="options">${q.options.map(o => `<li>${escapeHtml(o)}</li>`).join("")}</ul></div>`;
  }
  if (q.type === "tf") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><ul class="options"><li>A. 正确</li><li>B. 错误</li></ul></div>`;
  }
  if (q.type === "fill") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="blank-answer"></span> <span class="badge">${escapeHtml(q.difficulty)}</span></div></div>`;
  }
  return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><p class="muted">答题区：</p><p style="height:54px;border-bottom:1px solid var(--line)"></p></div>`;
}

function renderAnswer(q, idx) {
  return `<div class="answer-item"><strong>${idx}. [${escapeHtml(q.id)}] ${escapeHtml(q.answer)}</strong><p>${escapeHtml(q.explanation)}</p><span class="muted">来源：${escapeHtml(q.source)} · ${escapeHtml(q.topic)}</span></div>`;
}

function renderReview() {
  const r = DB.review;
  $("reviewBody").innerHTML = (r.sections || []).map(sec => `
    <section class="review-section">
      <h2>${escapeHtml(sec.title)}</h2>
      <ul>${(sec.points || []).map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>
      ${(sec.details || []).length ? `<h3>详细知识点</h3><ul>${sec.details.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${(sec.must_memorize || []).length ? `<h3>必背结论</h3><ul>${sec.must_memorize.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${(sec.formulas || []).map(f => `<div class="formula">${escapeHtml(f)}</div>`).join("")}
      ${(sec.exam_focus || []).length ? `<p><strong>常考点：</strong>${escapeHtml(sec.exam_focus.join("；"))}</p>` : ""}
    </section>
  `).join("");
}

function init() {
  document.querySelectorAll("nav button").forEach(btn => btn.addEventListener("click", () => showView(btn.dataset.view)));
  $("searchInput").addEventListener("input", e => runSearch(e.target.value));
  $("makeStandardQuiz").addEventListener("click", () => generateStandardQuiz($("searchInput").value.trim()));
  $("makeBeginnerQuiz").addEventListener("click", () => generateBeginnerQuiz($("searchInput").value.trim()));
  $("quizAll").addEventListener("click", () => generateStandardQuiz(""));
  $("beginnerAll").addEventListener("click", () => generateBeginnerQuiz(""));
  renderStats();
  renderTermCloud();
  renderOutline();
  renderMindmaps();
  renderReview();
  runSearch("");
  generateBeginnerQuiz("", false);
}

init();
