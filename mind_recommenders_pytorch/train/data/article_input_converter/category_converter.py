import torch


class CategoryConverter:
    def __init__(
        self,
        category_file_path=None,
        subcategory_file_path=None,
    ):
        if category_file_path:
            self.category_serializer = CategorySerializer(category_file_path)
        else:
            self.convert_category = lambda *_, **__: self._raise_exception_for_missing_argument("category")

        if subcategory_file_path:
            self.subcategory_serializer = CategorySerializer(subcategory_file_path)
        else:
            self.convert_subcategory = lambda *_, **__: self._raise_exception_for_missing_argument("subcategory")

    def convert_category(self, articles):
        return torch.IntTensor([self.category_serializer.run(category) for category in articles.category.values])

    def convert_subcategory(self, articles):
        return torch.IntTensor([self.subcategory_serializer.run(category) for category in articles.subcategory.values])

    @staticmethod
    def _raise_exception_for_missing_argument(category_name):
        raise RuntimeError(f"argument {category_name}_file_path is not specified at initialization.")


class CategorySerializer:
    """カテゴリをintに変換する（未知のカテゴリの場合は0をreturn）"""

    def __init__(self, category_file_path):
        self.categories = [x.rstrip() for x in open(category_file_path)]
        self.category2idx = dict(zip(self.categories, range(1, len(self.categories) + 1)))

    def run(self, category):
        return self.category2idx.get(category, 0)
