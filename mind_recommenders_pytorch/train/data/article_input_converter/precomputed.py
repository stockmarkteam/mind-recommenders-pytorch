from ..utils import stack_article_inputs

class ArticleInputConverter:
    def __init__(self):
        pass
    def run(self, articles, _):
        return stack_article_inputs(articles.article_input.values)
