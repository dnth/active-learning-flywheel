from mkdocs_gen_files import EditableFile

with open("../../README.md", "r") as f:
    readme_content = f.read()

with open("../index.md", "w") as f:
    f.write(readme_content)