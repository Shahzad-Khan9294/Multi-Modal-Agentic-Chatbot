import os
import redis

class RedisClient:
    def __init__(self):
        self.redis_conn = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True)
    def set(self, key, value):
        self.redis_conn.set(key, value)
    def get(self, key):
        return self.redis_conn.get(key)
    def raw(self):
        return self.redis_conn
redis_conn = RedisClient()