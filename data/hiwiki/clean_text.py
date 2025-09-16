import os
import glob
from concurrent.futures import ThreadPoolExecutor

import regex
import pypandoc

START_TOKEN = "[START]"
END_TOKEN = "[END]"

def clean_text(read_path: str, write_path: str):
    content: str
    print("Reading File:", read_path)
    with open(read_path, 'r', encoding="utf-8") as f:
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
        f'{START_TOKEN}\n\n<h1>' + r'\1' + '</h1>', content
    )
    content = regex.sub(
        r"</doc>",
        END_TOKEN, content
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

    split_docs = content.split(f"\n{END_TOKEN}\n{START_TOKEN}\n\n")
    split_docs[0] = split_docs[0][len(f"{START_TOKEN}\n"):]
    split_docs[-1] = split_docs[-1][:-len(f"\n{END_TOKEN}")]

    for i in range(len(split_docs)):
        split_docs[i] = pypandoc.convert_text(
            split_docs[i],
            to='md',
            format='html',
            extra_args=["--wrap=none", "-M2GB", "+RTS", "-K64m", "-RTS"]
        )

    split_docs[0] = f"{START_TOKEN}\n\n" + split_docs[0]
    split_docs[-1] = split_docs[-1] + f"\n{END_TOKEN}"
    content = f"\n{END_TOKEN}\n{START_TOKEN}\n\n".join(split_docs)

    print("Writing File:", write_path)
    if not os.path.exists(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path))
    with open(write_path, 'w', encoding='utf-8') as f:
        f.write(content)

data_dir = "./texthtml"
data_paths = glob.glob(os.path.join(data_dir, "**"), recursive=True)
data_files = [item_path for item_path in data_paths if os.path.isfile(item_path)]

out_dir = "./textmd"
out_files = [os.path.join(out_dir, file_path[len(data_dir)+1:]) for file_path in data_files]

with ThreadPoolExecutor() as executor:
    executor.map(clean_text, data_files, out_files)
