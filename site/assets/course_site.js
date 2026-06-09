const DB = window.COURSE_DB;
const $ = (id) => document.getElementById(id);
const norm = (s) => (s || "").toString().toLowerCase();
const escapeHtml = (s) => (s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const lastQuizState = { beginner: null, standard: null };

function queueMathTypeset(root = document.body, attempt = 0) {
  const run = () => {
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise([root]).catch(err => console.warn("MathJax typeset failed", err));
    } else if (attempt < 30) {
      window.setTimeout(() => queueMathTypeset(root, attempt + 1), 120);
    }
  };
  if (window.MathJax && MathJax.startup && MathJax.startup.promise) {
    MathJax.startup.promise.then(run);
  } else {
    window.setTimeout(run, 120);
  }
}

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

function safeFileName(text) {
  return (text || "试卷").replace(/[\\/:*?"<>|]+/g, "-").replace(/\s+/g, "-").slice(0, 80);
}

function downloadTextFile(filename, text) {
  const blob = new Blob([text], {type: "text/markdown;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function printStyleBlock() {
  return `<style>
@media print {
  @page { margin: 7mm; }
  body { font-size: 10.5pt; line-height: 1.32; }
  h1, h2, h3 { page-break-after: avoid; margin: 0.45em 0 0.25em; }
  p, li { margin: 0.2em 0; }
  .blank { display: inline-block; min-width: 42mm; border-bottom: 1px solid #777; }
}
</style>`;
}

function questionMarkdown(q, idx) {
  if (q.type === "mcq") {
    return `${idx}. ${q.stem}\n\n${q.options.map(o => `   ${o}`).join("\n")}`;
  }
  if (q.type === "material") {
    const subs = q.subquestions || q.questions || [];
    return `${idx}. ${q.stem}\n\n材料：${q.material || ""}\n\n${subs.map((x, i) => `   ${i + 1}. ${x}`).join("\n")}\n\n答题区：\n\n\n`;
  }
  return `${idx}. ${q.stem}\n\n答题区：\n\n\n`;
}

function answerMarkdown(q, idx) {
  return `${idx}. [${q.id}] 答案：${q.answer}\n\n解析：${q.explanation}\n\n来源：${q.source} · ${q.topic}`;
}

function exportQuizMarkdown(mode, part) {
  const state = lastQuizState[mode];
  if (!state || !state.chosen.length) return;
  const modeName = mode === "beginner" ? "基础练习" : "标准组卷";
  const title = `${modeName}${state.topic ? "：" + state.topic : ""}`;
  const labels = {mcq: "单选题", short: "简答题", material: "资料题"};
  const lines = [
    printStyleBlock(),
    `# ${title}${part === "questions" ? "（题目版）" : "（答案解析版）"}`,
    "",
    `生成时间：${new Date().toLocaleString("zh-CN")}`,
    "",
    part === "questions" ? "> 打印建议：题目和答案分开打印；本文件已内置较小页边距样式。" : "> 打印建议：答案解析单独打印或仅在核对时查看。",
    "",
  ];
  if (part === "questions") {
    state.spec.forEach(([type]) => {
      const items = state.chosen.filter(q => q.type === type);
      if (!items.length) return;
      lines.push(`## ${labels[type]}`, "");
      items.forEach((q, i) => lines.push(questionMarkdown(q, i + 1), ""));
    });
  } else {
    state.chosen.forEach((q, i) => lines.push(answerMarkdown(q, i + 1), ""));
  }
  const suffix = part === "questions" ? "题目" : "答案解析";
  downloadTextFile(`${safeFileName(title)}-${suffix}.md`, lines.join("\n"));
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
  ` : `<h2>搜索说明</h2><p class="muted">搜索会定位到课件原文片段、相关词条和来源位置。适合先查概念，再回到教材目录和主题书库复习。</p>`;
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
    ? [["mcq", 10], ["short", 4], ["material", 1]]
    : [["mcq", 20], ["short", 6], ["material", 2]];
  const labels = {mcq: "单选题", short: "简答题", material: "资料题"};
  const chosen = spec.flatMap(([type, count]) => pickQuestions(bank, type, topic, count));
  const title = mode === "beginner" ? "基础练习随机题" : "标准综合随机题";
  const targetTitle = mode === "beginner" ? $("beginnerTitle") : $("quizTitle");
  const targetMeta = mode === "beginner" ? $("beginnerMeta") : $("quizMeta");
  const targetBody = mode === "beginner" ? $("beginnerBody") : $("quizBody");
  lastQuizState[mode] = {chosen, spec, topic, title};
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
  if (q.type === "material") {
    const subs = q.subquestions || q.questions || [];
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><p class="snippet"><strong>材料：</strong>${escapeHtml(q.material || "")}</p><ol>${subs.map(x => `<li>${escapeHtml(x)}</li>`).join("")}</ol><p class="muted">答题区：</p><p style="height:78px;border-bottom:1px solid var(--line)"></p></div>`;
  }
  return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><p class="muted">答题区：</p><p style="height:54px;border-bottom:1px solid var(--line)"></p></div>`;
}

function renderAnswer(q, idx) {
  return `<div class="answer-item"><strong>${idx}. [${escapeHtml(q.id)}] ${escapeHtml(q.answer)}</strong><p>${escapeHtml(q.explanation)}</p><span class="muted">来源：${escapeHtml(q.source)} · ${escapeHtml(q.topic)}</span></div>`;
}

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
}

function markdownToHtml(md) {
  const lines = (md || "").split(/\r?\n/);
  const html = [];
  let listOpen = false;
  let inCode = false;
  let inMath = false;
  let codeLines = [];
  let mathLines = [];
  const closeList = () => {
    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
  };
  lines.forEach(line => {
    const trimmed = line.trim();
    const singleLineMath = trimmed.match(/^\$\$(.+)\$\$$/);
    if (!inCode && singleLineMath) {
      closeList();
      html.push(`<div class="math-block">$$${escapeHtml(singleLineMath[1].trim())}$$</div>`);
      return;
    }
    if (!inCode && trimmed === "$$") {
      if (inMath) {
        html.push(`<div class="math-block">$$${escapeHtml(mathLines.join("\n"))}$$</div>`);
        mathLines = [];
        inMath = false;
      } else {
        closeList();
        inMath = true;
      }
      return;
    }
    if (inMath) {
      mathLines.push(line);
      return;
    }
    if (line.trim().startsWith("```")) {
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCode = false;
      } else {
        closeList();
        inCode = true;
      }
      return;
    }
    if (inCode) {
      codeLines.push(line);
      return;
    }
    if (!line.trim()) {
      closeList();
      return;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 1, 5);
      const id = safeFileName(heading[2]).toLowerCase();
      html.push(`<h${level} id="${escapeHtml(id)}">${inlineMarkdown(heading[2])}</h${level}>`);
      return;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${inlineMarkdown(line.replace(/^\s*[-*]\s+/, ""))}</li>`);
      return;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      closeList();
      html.push(`<p>${inlineMarkdown(line)}</p>`);
      return;
    }
    if (/^\|.+\|$/.test(line)) {
      closeList();
      html.push(`<p class="table-line">${inlineMarkdown(line)}</p>`);
      return;
    }
    closeList();
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  });
  closeList();
  if (inMath) html.push(`<div class="math-block">$$${escapeHtml(mathLines.join("\n"))}$$</div>`);
  if (inCode) html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  return html.join("");
}

function renderTopicGuideNav(items) {
  return items.map((g, i) => `
    <button class="book-nav-item ${i === 0 ? "active" : ""}" data-guide="${escapeHtml(g.id)}">
      <strong>${escapeHtml(g.label)}</strong>
      <span>${g.exists ? `${g.wordCount} 字` : "待生成"}</span>
    </button>
  `).join("");
}

function renderTopicGuideContent(guide) {
  if (!guide || !guide.exists) {
    return `<div class="panel"><h2>讲义还在生成</h2><p class="muted">这个主题的 Markdown 文件尚未写入，完成后会自动显示在这里。</p></div>`;
  }
  const toc = (guide.headings || []).filter(h => h.level <= 3).slice(0, 28);
  return `
    <article class="book-reader">
      <div class="book-toolbar">
        <div>
          <h2>${escapeHtml(guide.title)}</h2>
          <p class="muted">${guide.wordCount} 字 · Markdown 讲义 · 可打印</p>
        </div>
        <div class="quiz-actions">
          <a class="btn secondary" href="${escapeHtml(guide.href)}" download>下载 Markdown</a>
          <button class="btn secondary" data-print-guide>打印本讲义</button>
        </div>
      </div>
      ${(guide.tocImages || []).length ? `
        <div class="toc-image-strip">
          ${guide.tocImages.map(img => `
            <figure>
              <a href="${escapeHtml(img)}" target="_blank"><img src="${escapeHtml(img)}" alt="${escapeHtml(guide.title)} 书本式目录图"></a>
              <figcaption><a href="${escapeHtml(img)}" download>下载目录图</a></figcaption>
            </figure>
          `).join("")}
        </div>
      ` : ""}
      <div class="book-layout">
        <aside class="book-toc">
          <h3>本讲义目录</h3>
          ${toc.map(h => `<a class="toc-level-${h.level}" href="#${escapeHtml(safeFileName(h.title).toLowerCase())}">${escapeHtml(h.title)}</a>`).join("")}
        </aside>
        <div class="markdown-body">${markdownToHtml(guide.content)}</div>
      </div>
    </article>
  `;
}

function renderReview() {
  const library = DB.topicGuides || {items: []};
  const items = library.items || [];
  $("reviewBody").innerHTML = `
    <div class="book-shell">
      <aside class="panel book-sidebar">
        <h3>主题书库</h3>
        <p class="muted">按学习主题重新组织，不再按课件文件硬拆。</p>
        <div id="topicGuideNav">${renderTopicGuideNav(items)}</div>
      </aside>
      <div id="topicGuideReader">${renderTopicGuideContent(items[0])}</div>
    </div>
  `;
  queueMathTypeset($("topicGuideReader"));
  $("topicGuideNav").querySelectorAll("button[data-guide]").forEach(btn => btn.addEventListener("click", () => {
    $("topicGuideNav").querySelectorAll("button").forEach(b => b.classList.toggle("active", b === btn));
    const guide = items.find(g => g.id === btn.dataset.guide);
    $("topicGuideReader").innerHTML = renderTopicGuideContent(guide);
    const printBtn = $("topicGuideReader").querySelector("[data-print-guide]");
    if (printBtn) printBtn.addEventListener("click", () => window.print());
    queueMathTypeset($("topicGuideReader"));
  }));
  const printBtn = $("topicGuideReader").querySelector("[data-print-guide]");
  if (printBtn) printBtn.addEventListener("click", () => window.print());
}

function init() {
  document.querySelectorAll("nav button").forEach(btn => btn.addEventListener("click", () => showView(btn.dataset.view)));
  $("searchInput").addEventListener("input", e => runSearch(e.target.value));
  $("makeStandardQuiz").addEventListener("click", () => generateStandardQuiz($("searchInput").value.trim()));
  $("makeBeginnerQuiz").addEventListener("click", () => generateBeginnerQuiz($("searchInput").value.trim()));
  $("quizAll").addEventListener("click", () => generateStandardQuiz(""));
  $("beginnerAll").addEventListener("click", () => generateBeginnerQuiz(""));
  $("quizExportQuestions").addEventListener("click", () => exportQuizMarkdown("standard", "questions"));
  $("quizExportAnswers").addEventListener("click", () => exportQuizMarkdown("standard", "answers"));
  $("beginnerExportQuestions").addEventListener("click", () => exportQuizMarkdown("beginner", "questions"));
  $("beginnerExportAnswers").addEventListener("click", () => exportQuizMarkdown("beginner", "answers"));
  renderStats();
  renderTermCloud();
  renderOutline();
  renderReview();
  runSearch("");
  generateBeginnerQuiz("", false);
}

init();
