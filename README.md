# 深度学习课件复习网站

打开网站：

- `site/index.html`

当前保留内容：

- `database/`：从课件抽取出的可搜索数据库、词条索引、复习资料和题库。
- `site/`：可直接打开的静态网站。
- `scripts/build_review_site.py`：重新生成数据库和网站的脚本。

之后新增课件时，把新的 `.ipynb` 放到上一级课件目录，然后运行：

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

说明：

- 本工程没有保留零散转换 `.py` 文件。
- 原始课件未被修改。
- 当前已纳入 13 个课件文件。
- 当前题库共 575 道：289 道单选、146 道填空、140 道简答。
- 网站每次都会随机抽取 20 道单选、10 道填空、5 道简答，并在最后给出解析与来源。
- 以当前题量，足够支持至少 10 套完全不同的整卷随机生成。
- 复习资料已按考试用途补充“详细知识点、必背结论、公式拆解、命题方式、答题模板、易错陷阱”和参考资料。
- 新增课件 `注意力机制及Transformer1-学生分发版-终版2026.ipynb` 已补入搜索、题库和第 8 章复习资料。
