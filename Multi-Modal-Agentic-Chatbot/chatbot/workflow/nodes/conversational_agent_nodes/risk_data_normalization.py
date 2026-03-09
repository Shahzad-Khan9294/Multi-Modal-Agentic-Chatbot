import json
import logging
from datetime import datetime
logger = logging.getLogger(__name__)
from collections import defaultdict
from dateutil.parser import parse as parse_date


class RiskCategoryDataNormalizatoion:
    def dedupe_per_day(self, rows):
        per_day = defaultdict(list)
        for r in rows:
            per_day[r["date"]].append(r)
        deduped = [
            max(records, key=lambda x: x["timestamp"])
            for records in per_day.values()]
        deduped.sort(key=lambda x: x["timestamp"])
        return deduped

    def pick_last_record_per_day(self, rows):
        logger.info("--------- DAILY NORMALIZATION (<= 1 Month) ---------")
        return self.dedupe_per_day(rows)

    def pick_monthly_samples(self, rows):
        rows = self.dedupe_per_day(rows)
        per_month = defaultdict(list)
        for r in rows:
            per_month[r["month"]].append(r)
        selected = []
        for month, records in per_month.items():
            records.sort(key=lambda x: x["timestamp"])
            n = len(records)
            if n <= 10:
                selected.extend(records)
                continue
            start = records[:3]
            mid_start = max(0, n // 2 - 1)
            middle = records[mid_start: mid_start + 3]
            end = records[-3:]
            unique = {r["timestamp"]: r for r in start + middle + end}
            selected.extend(unique.values())
        logger.info("--------- MONTHLY NORMALIZATION (>1 Month) ---------")
        selected.sort(key=lambda x: x["timestamp"])
        return selected

    def normalize_rows(self, rows, category_description=None):
        normalized = []
        for row in rows:
            if isinstance(row, str):
                payload = json.loads(row)
            elif isinstance(row, dict):
                payload = row
            else:
                payload = dict(row._mapping)
            if category_description:
                payload["risk_category_description"] = category_description
            ts_raw = (
                payload.get("date_time")
                or payload.get("timestamp")
                or datetime.utcnow())
            if isinstance(ts_raw, str):
                ts = parse_date(ts_raw)
            else:
                ts = ts_raw
            normalized.append({
                "timestamp": ts,
                "date": ts.date(),
                "month": (ts.year, ts.month),
                "json": json.dumps(payload, default=str)})
        return normalized

    def select_records_by_timerange(self, rows, start_date, end_date):
        days = (end_date - start_date).days
        logger.info(f"---------- RISK NORM NO OF DAYS --------: {days}")
        if days <= 23:
            return self.pick_last_record_per_day(rows)
        else:
            return self.pick_monthly_samples(rows)
rcdn = RiskCategoryDataNormalizatoion()


