# Download Data
wget https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2

# Extract Data
if ! command -v lbzip2 &> /dev/null; then
    sudo apt update
    sudo apt install -y lbzip2
fi
lbzip2 -dkv hiwiki-latest-pages-articles.xml.bz2

# Convert to HTML Format
pip3 install wikiextractor
python3 -m wikiextractor.WikiExtractor hiwiki-latest-pages-articles.xml --html --links

# Clean the Format to produce Markdown
if ! command -v pandoc &> /dev/null; then
    sudo apt update
    sudo apt install -y pandoc
fi
pip3 install pypandoc
python3 clean_text.py
