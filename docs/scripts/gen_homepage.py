import mkdocs_gen_files

with mkdocs_gen_files.open("../../README.md", "r") as f:
    readme_content = f.read()

with mkdocs_gen_files.open("../index.md", "w") as f:
    f.write(readme_content)
