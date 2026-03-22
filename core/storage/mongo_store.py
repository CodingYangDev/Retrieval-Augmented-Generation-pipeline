# =========================================
# MongoDB（存储完整文档）
# 用于：父子块回溯
# =========================================

from pymongo import MongoClient
from config.settings import settings


class MongoStore:

    def __init__(self):
        """
        初始化 Mongo 连接
        """
        self.client = MongoClient(settings.MONGO_URI)
        self.db = self.client[settings.DB_NAME]
        self.col = self.db["documents"]

    def save_parent(self, parent_id, text):
        """
        存储完整文档
        """
        self.col.insert_one({
            "_id": parent_id,
            "text": text
        })

    def get_parent(self, parent_id):
        """
        获取完整文档
        """
        doc = self.col.find_one({"_id": parent_id})
        return doc["text"] if doc else ""