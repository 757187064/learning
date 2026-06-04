# 深度学习期末复习资料库

打开网站：

- `site/index.html`
- GitHub Pages: https://757187064.github.io/learning/

当前保留内容：

- `database/`：从课件抽取出的搜索数据库、教材目录、词条索引、基础题库、标准题库、思维导图索引。
- `site/`：可直接打开或由 GitHub Pages 发布的静态网站。
- `site/mindmaps/`：每个课件的总览 SVG 思维导图，以及章节 SVG/Markdown 大纲下载文件。
- `site/guides/`：按课件生成的 Markdown 考试复习讲义，以及总目录。
- `scripts/build_review_site.py`：重新生成数据库和网站的入口脚本。
- `scripts/build_review_site_v2.py`：当前新版构建逻辑。

之后新增课件时，把新的 `.ipynb` 或 `.pptx` 放到上一级课件目录，然后运行：

```bash
python3 scripts/build_review_site.py
```

接入 Git：

```bash
cd "/Users/sakiko/Public/deeplearing/课件和笔记/课件/深度学习复习网站"
git init -b main
git add .
git commit -m "Initial commit for review site"
```

如果之后要接 GitHub 远端：

```bash
git remote add origin <your-repo-url>
git push -u origin main
```

当前仓库一键发布：

```bash
bash scripts/publish_to_github.sh git@github.com:757187064/learning.git
```

GitHub Pages 自动发布：

- 仓库里已经包含 `.github/workflows/deploy-pages.yml`，推送到 `main` 后会自动把 `site/` 目录发布为网页。
- 如果 GitHub 第一次提示你选择 Pages 来源，在仓库 `Settings -> Pages` 中选择 `GitHub Actions` 即可。

说明：

- 本工程没有保留零散转换 `.py` 文件。
- 原始课件未被修改。
- 当前已纳入 17 个课件文件，包括新增的 2 份图学习 PPT。
- 当前数据库包含 1026 个可检索片段、113 个索引词条。
- 标准题库保留 540 道题，已过滤机械模板题。
- 基础题库包含 2066 道低难度题，用于先熟悉课程脉络。
- 网站包含：搜索、教材目录、思维导图、基础练习、标准组卷、综合资料。
- 基础练习和标准组卷都支持分别导出“题目 Markdown”和“答案 Markdown”，导出文件内置小边距打印样式，适合节省纸张。
- 综合资料页提供 17 份按课件生成的 Markdown 考试复习讲义下载，以及讲义总目录。
- 思维导图先展示整份课件总览，章节小图折叠展示，并提供 SVG 与 Markdown 大纲下载。
- 构建脚本支持未来 `.pptx` 课件文本提取；图片型 PPT 会尝试调用 macOS Vision OCR，并用文件哈希缓存跳过未变化课件。
