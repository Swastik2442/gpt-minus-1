import os
import glob
import regex
import pypandoc

data_dir = "./text"
data_paths = glob.glob(os.path.join(data_dir, "**"), recursive=True)
data_files = [item_path for item_path in data_paths if os.path.isfile(item_path)]

for file_path in data_files:
    with open(file_path, 'r', encoding="utf-8") as f:
        content = f.read()

        # Clean Text
        content = regex.sub(
            r'&lt;ns&gt;.*?&lt;/ns&gt;\n|&lt;id&gt;.*?&lt;/id&gt;\n|&lt;revision&gt;\n|&lt;id&gt;.*?&lt;/id&gt;\n|&lt;parentid&gt;.*?&lt;/parentid&gt;\n|&lt;timestamp&gt;.*?&lt;/timestamp&gt;\n|&lt;contributor&gt;\n|&lt;username&gt;.*?&lt;/username&gt;\n|&lt;id&gt;.*?&lt;/id&gt;\n|&lt;/contributor&gt;\n|&lt;comment&gt;.*?&lt;/comment&gt;\n|&lt;origin&gt;.*?&lt;/origin&gt;\n|&lt;model&gt;.*?&lt;/model&gt;\n|&lt;format&gt;.*?&lt;/format&gt;\n|## [^\n]+\n|&lt;templatestyles src=\"[^\"]+\"[ ]*/&gt;\n',
            '', content
        )
        content = regex.sub(
            r'</<li>',
            r'</li>', content
        )

        # Add Start and End Tokens
        content = regex.sub(
            r"<doc id=\"\d+\" url=\"[^\"]+\" title=\"[^\"]+\">\n(.*?)\n",
            '[START]\n\n<h1>' + r'\1' + '</h1>', content
        )
        content = regex.sub(
            r"</doc>",
            '[END]', content
        )


        # Remove Links
        content = regex.sub(
            r'&lt;a\s+href="[^"]*"&gt;(.*?)&lt;/a&gt;',
            r'\1', content
        )

        # Parse HTML
        content = regex.sub(
            r'&lt;br&gt;',
            '\n', content
        )

        split_docs = content.split("\n[END]\n[START]\n\n")
        for i in range(len(split_docs)):
            split_docs[i] = pypandoc.convert_text(split_docs[i], to='md', format='html', extra_args=["-M2GB", "+RTS", "-K64m", "-RTS"])
        content = "\n[END]\n[START]\n\n".join(split_docs)
        print(content)
    break
