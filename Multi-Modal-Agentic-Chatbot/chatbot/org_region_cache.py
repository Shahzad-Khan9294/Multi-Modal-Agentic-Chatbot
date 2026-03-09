import asyncio
from chatbot.db import db


org_region_ids_cache = None
lock = asyncio.Lock()
async def init_org_region_cache(org_region_id: str):
    global org_region_ids_cache
    async with lock:
        if org_region_ids_cache is None:
            org_region_ids_cache = [
                db.get_org_region_document(org_region_id),
                db.get_org_region_event(org_region_id),
                db.get_org_region_client(org_region_id),
                db.get_org_region_stakeholder(org_region_id),
            ]

def get_org_region_cache():
    if org_region_ids_cache is None:
        raise RuntimeError("Org region cache not initialized")
    return org_region_ids_cache
