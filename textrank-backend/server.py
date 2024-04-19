from flask import Flask, request
from flask_cors import CORS

import text_rank
import string
import re

app = Flask(__name__)
CORS(app)

tokenizer = text_rank.ArticleTokenizer()

@app.route('/run/<int:co_occurances>/<int:directed>/<int:max_keywords>/<int:max_keyword_length>')
def run_script(co_occurances, directed, max_keywords, max_keyword_length):
    article_text = re.sub(r'\s+', ' ', request.args.get('text'))
    print(request.args.get('text'))

    candidate_keywords = tokenizer.extract_keywords(article_text, 1)
    words_list = tokenizer.extract_words(article_text)

    word_graph = text_rank.WordGraph(candidate_keywords, words_list, max_distance=co_occurances, directed=directed)
    word_graph.calculate_pagerank()

    keywords = [keyword.capitalize() for keyword in word_graph.postprocess_keywords_chronological(max_keywords, max_keyword_length) if bool(keyword)]
    tagged_keywords = tokenizer.tag_keywords(keywords)

    keywords = list(set([tagged_keyword[0] for tagged_keyword in tagged_keywords if (tagged_keyword[1][0] == 'N' or tagged_keyword[1][0] == 'J') and len(tagged_keyword[0]) > 1]))

    return keywords

if __name__ == '__main__':
    app.run(debug=True)
