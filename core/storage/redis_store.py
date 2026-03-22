# =========================================
# Redis 缓存（提升QPS）
# =========================================

import redis
from config.settings import settings


class RedisStore:

    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True  # 自动转字符串
        )

    def get(self, key):
        """
        获取缓存
        """
        return self.client.get(key)

    def set(self, key, value, ttl=3600):
        """
        写入缓存
        """
        self.client.set(key, value, ex=ttl)