# -*- coding: utf-8 -*-
"""
Module thu thập bình luận Facebook thông qua Bright Data Datasets Trigger API.
- Hỗ trợ xoay tua danh sách nhiều Bright Data API keys từ file bright_data_api_key.txt.
- Tự động phát hiện và loại bỏ key chết/hết credit, chuyển ngay sang key kế tiếp.
- Giữ nguyên tiếng Việt có dấu chuẩn UTF-8 (fix lỗi encoding của requests).
- Chuẩn hóa cấu trúc đọc ghi append CSV an toàn.
"""

import argparse
import csv
import io
import logging
import os
import sys
import time
import requests
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSING_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(DATA_PROCESSING_DIR)

# Tự động nạp cấu hình từ .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY_FILE = os.path.join(CURRENT_DIR, "bright_data_api_key.txt")
LOG_FILE = os.path.join(CURRENT_DIR, "log.txt")
POSTS_CSV_FILE = os.path.join(DATA_PROCESSING_DIR, "crawl_fb_post_by_serp", "facebook_car_posts.csv")
DEFAULT_OUTPUT_CSV = os.path.join(CURRENT_DIR, "facebook_car_comments.csv")

BRIGHT_DATA_PROGRESS_URL = "https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
BRIGHT_DATA_SNAPSHOT_URL = "https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=csv"

logger = logging.getLogger("crawl_fb_comment")
logger.setLevel(logging.INFO)

if not logger.handlers:
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(logging.Formatter("%(message)s"))
	logger.addHandler(console_handler)

	file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
	file_handler.setLevel(logging.ERROR)
	file_handler.setFormatter(
		logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
	)
	logger.addHandler(file_handler)


def load_api_keys(file_path=API_KEY_FILE):
	"""
	Đọc danh sách các cặp (api_key, dataset_id) từ biến môi trường (.env) hoặc file bright_data_api_key.txt.
	Lưu ý: api_key và dataset_id luôn đi cùng nhau theo từng cặp.
	"""
	credentials = []

	# 1. Đọc các cặp từ biến môi trường BRIGHT_DATA_PAIRS (dạng: key1:dataset1,key2:dataset2,...)
	env_pairs = os.getenv("BRIGHT_DATA_PAIRS", "").strip()
	if env_pairs:
		for item in env_pairs.split(","):
			item = item.strip()
			if not item:
				continue
			if ":" in item:
				k, d = item.split(":", 1)
				credentials.append({"api_key": k.strip(), "dataset_id": d.strip()})
			else:
				logger.warning(f"Bỏ qua cặp key không hợp lệ (thiếu dấu ':' giữa api_key và dataset_id): {item}")
		if credentials:
			logger.info(f"[*] Đã nạp thành công {len(credentials)} cặp (api_key, dataset_id) từ biến môi trường (.env)")
			return credentials

	# 2. Đọc dự phòng từ file text nếu không có trong .env
	if os.path.exists(file_path):
		logger.warning(f"[!] Không tìm thấy BRIGHT_DATA_PAIRS trong .env. Đang đọc dự phòng từ: {file_path}")
		with open(file_path, "r", encoding="utf-8-sig") as f:
			reader = csv.reader(f)
			for row in reader:
				if not row:
					continue
				cleaned = [col.strip().strip('"').strip("'") for col in row if col.strip()]
				if not cleaned:
					continue

				first_col = cleaned[0]
				if first_col.startswith("#") or first_col.lower() in ["api_key", "key"]:
					continue

				if len(cleaned) >= 2:
					api_key = cleaned[0]
					dataset_id = cleaned[1]
					credentials.append({"api_key": api_key, "dataset_id": dataset_id})
				else:
					logger.warning(f"Bỏ qua dòng không có đủ api_key và dataset_id: {row}")

		if credentials:
			return credentials

	logger.error(f"Không tìm thấy Bright Data credentials trong biến môi trường (.env) hoặc file text: {file_path}")
	return []


class BrightDataKeyManager:
	def __init__(self, credentials):
		self.credentials = list(credentials)
		self.current_idx = 0

	def get_current(self):
		if not self.credentials:
			return None
		return self.credentials[self.current_idx]

	def get_current_key(self):
		curr = self.get_current()
		return curr["api_key"] if curr else None

	def get_current_dataset_id(self):
		curr = self.get_current()
		return curr["dataset_id"] if curr else None

	def rotate_key(self):
		if not self.credentials:
			return None
		self.current_idx = (self.current_idx + 1) % len(self.credentials)
		curr = self.get_current()
		logger.info(
			f"[*] [Key Manager] Xoay tua sang Key [{self.current_idx + 1}/{len(self.credentials)}] "
			f"(Key: ...{curr['api_key'][-8:]} | Dataset: {curr['dataset_id']})"
		)
		return curr

	def remove_current_and_switch(self):
		if not self.credentials:
			return None
		bad_cred = self.credentials.pop(self.current_idx)
		logger.warning(
			f"[-] [Key Manager] Đã loại bỏ Key lỗi: ...{bad_cred['api_key'][-8:]} (Dataset: {bad_cred['dataset_id']}). Còn lại {len(self.credentials)} keys."
		)
		if not self.credentials:
			logger.critical("[!] Đã dùng hết toàn bộ danh sách API Keys!")
			return None
		self.current_idx = self.current_idx % len(self.credentials)
		return self.get_current()


def load_post_urls(csv_file=POSTS_CSV_FILE):
	if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
		logger.error(f"File bài viết '{csv_file}' không tồn tại hoặc rỗng!")
		return []

	urls = []
	seen = set()
	try:
		with open(csv_file, mode="r", encoding="utf-8-sig") as f:
			for row in csv.DictReader(f):
				u = row.get("url", "").strip()
				if u and u not in seen:
					seen.add(u)
					urls.append(u)
	except Exception as e:
		logger.error(f"Lỗi khi đọc file CSV bài viết: {e}")
		return []

	return urls


def trigger_scrape_batch(key_manager, urls, limit_records=20, get_all_replies=False, comments_sort="Most relevant"):
	payload = [
		{
			"url": u,
			"get_all_replies": get_all_replies,
			"limit_records": limit_records,
			"comments_sort": comments_sort
		}
		for u in urls
	]

	while True:
		curr = key_manager.get_current()
		if not curr:
			return None, None

		api_key = curr["api_key"]
		dataset_id = curr["dataset_id"]
		trigger_url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={dataset_id}&notify=false&include_errors=true"

		headers = {
			"Authorization": f"Bearer {api_key}",
			"Content-Type": "application/json"
		}

		try:
			res = requests.post(trigger_url, headers=headers, json=payload, timeout=60)

			# Bắt lỗi tài khoản hết hạn ngạch hoặc inactive
			if res.status_code in [401, 403, 429] or (res.status_code == 400 and (
					"customer is not active" in res.text.lower() or "quota" in res.text.lower() or "credit" in res.text.lower()
			)):
				logger.warning(
					f"[-] Key ...{api_key[-8:]} (Dataset: {dataset_id}) bị từ chối (HTTP {res.status_code}: {res.text[:100]}). Đổi key...")
				next_cred = key_manager.remove_current_and_switch()
				if not next_cred:
					return None, None
				continue

			if res.status_code not in [200, 201, 202]:
				logger.error(f"Lỗi Trigger API (HTTP {res.status_code}): {res.text[:200]}")
				return None, api_key

			data = res.json()
			if isinstance(data, dict):
				snapshot_id = data.get("snapshot_id")
				if snapshot_id:
					return snapshot_id, api_key

			logger.error(f"Phản hồi không chứa snapshot_id: {res.text[:150]}")
			return None, api_key

		except Exception as e:
			logger.error(f"Lỗi kết nối khi gửi Trigger: {e}")
			return None, api_key


def wait_and_download_csv(api_key, snapshot_id, poll_interval=10, max_wait_sec=600):
	headers = {"Authorization": f"Bearer {api_key}"}
	progress_url = BRIGHT_DATA_PROGRESS_URL.format(snapshot_id=snapshot_id)
	snapshot_url = BRIGHT_DATA_SNAPSHOT_URL.format(snapshot_id=snapshot_id)

	logger.info(f"  [*] Đang theo dõi Snapshot ID: {snapshot_id}")
	start_time = time.time()

	while time.time() - start_time < max_wait_sec:
		try:
			p_res = requests.get(progress_url, headers=headers, timeout=30)
			if p_res.status_code == 200:
				p_data = p_res.json()
				status = p_data.get("status", "").lower()
				records_count = p_data.get("records", 0)

				if status == "ready":
					logger.info(f"  [✔] Snapshot READY! Cào được: {records_count} records")
					d_res = requests.get(snapshot_url, headers=headers, timeout=90)
					if d_res.status_code == 200:
						# Ép chuẩn utf-8 để giữ dấu tiếng Việt không bị lỗi font
						return d_res.content.decode("utf-8", errors="replace")
					else:
						logger.error(f"  [!] Lỗi tải CSV: HTTP {d_res.status_code}")
						return None

				if status in ["failed", "canceled"]:
					logger.error(f"  [!] Snapshot kết thúc với trạng thái: {status}")
					return None

				logger.info(f"      -> Trạng thái: {status}... đợi {poll_interval}s")
			else:
				logger.warning(f"      [!] Check progress HTTP {p_res.status_code}")

		except Exception as e:
			logger.warning(f"      [!] Lỗi mạng khi check progress: {e}")

		time.sleep(poll_interval)

	logger.error(f"  [!] Quá thời gian chờ {max_wait_sec}s cho Snapshot {snapshot_id}")
	return None


def append_csv_text_to_file(target_file, new_csv_text):
	if not new_csv_text or not new_csv_text.strip():
		return 0

	file_exists = os.path.exists(target_file) and os.path.getsize(target_file) > 0
	reader = csv.reader(io.StringIO(new_csv_text))

	try:
		header = next(reader)
	except StopIteration:
		return 0

	comment_idx = header.index("comment_text") if "comment_text" in header else None
	url_idx = header.index("url") if "url" in header else None

	valid_rows = []
	for row in reader:
		if not row:
			continue
		if comment_idx is not None and (comment_idx >= len(row) or not row[comment_idx].strip()):
			continue
		if url_idx is not None and (url_idx >= len(row) or not row[url_idx].strip()):
			continue
		valid_rows.append(row)

	if not valid_rows:
		return 0

	with open(target_file, mode="a", encoding="utf-8-sig", newline="") as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(header)
		writer.writerows(valid_rows)

	return len(valid_rows)


def crawl_facebook_comments(input_file=POSTS_CSV_FILE, output_file=DEFAULT_OUTPUT_CSV, batch_size=20, limit_per_post=20, max_posts=None):
	logger.info("=" * 70)
	logger.info("[*] BẮT ĐẦU CÀO BÌNH LUẬN FACEBOOK (XOAY TUA API KEYS)")
	logger.info("=" * 70)

	keys = load_api_keys(API_KEY_FILE)
	if not keys:
		logger.error("Dừng chương trình do không có API key khả dụng!")
		return

	key_manager = BrightDataKeyManager(keys)
	urls = load_post_urls(input_file)
	if not urls:
		logger.error("Dừng chương trình: không có URL bài viết!")
		return

	if max_posts and len(urls) > max_posts:
		urls = urls[:max_posts]

	total_posts = len(urls)
	total_batches = (total_posts + batch_size - 1) // batch_size

	logger.info(f"[*] Tổng URL xử lý: {total_posts}")
	logger.info(f"[*] Số lượng API Keys sẵn sàng: {len(keys)}")
	logger.info(f"[*] Cấu hình: {batch_size} URLs/batch | Tối đa {limit_per_post} cmt/URL")
	logger.info(f"[*] File xuất kết quả: {output_file}")
	logger.info("=" * 70)

	total_downloaded = 0
	start_time = time.time()

	for b_idx in range(total_batches):
		batch = urls[b_idx * batch_size: (b_idx + 1) * batch_size]
		current_k = key_manager.get_current_key()
		current_ds = key_manager.get_current_dataset_id()
		if not current_k:
			logger.error("[!] Đã hết toàn bộ API key, buộc phải dừng sớm.")
			break

		logger.info(f"\n[Batch {b_idx + 1}/{total_batches}] Gửi {len(batch)} URLs bằng Key ...{current_k[-8:]} (Dataset: {current_ds})")

		snapshot_id, used_key = trigger_scrape_batch(
			key_manager=key_manager,
			urls=batch,
			limit_records=limit_per_post,
			get_all_replies=False,
			comments_sort="Most relevant"
		)

		if not snapshot_id:
			logger.error(f"Khởi tạo Snapshot thất bại cho Batch {b_idx + 1}, tiếp tục batch sau...")
			continue

		csv_content = wait_and_download_csv(used_key, snapshot_id)
		if csv_content:
			added = append_csv_text_to_file(output_file, csv_content)
			total_downloaded += added
			logger.info(f"  -> Đã gộp +{added} comments vào file. Tổng tích lũy: {total_downloaded}")
		else:
			logger.warning(f"  [!] Không lấy được CSV từ snapshot {snapshot_id}")

		if key_manager.get_current_key() == used_key:
			key_manager.rotate_key()

		time.sleep(2)

	elapsed = time.time() - start_time
	logger.info("\n" + "=" * 70)
	logger.info("[✔] HOÀN TẤT THU THẬP BÌNH LUẬN")
	logger.info(f"[*] Tổng số bản ghi bình luận đã thu thập: {total_downloaded}")
	logger.info(f"[*] Thời gian thực thi: {elapsed:.1f} giây")
	logger.info(f"[*] File kết quả: {output_file}")
	logger.info("=" * 70)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Cào bình luận Facebook bằng Bright Data Trigger API (Rotate Keys)")
	parser.add_argument("--input", type=str, default=POSTS_CSV_FILE, help="Đường dẫn file CSV chứa URLs bài viết")
	parser.add_argument("--batch-size", type=int, default=20, help="Số URL gửi mỗi batch (mặc định 20)")
	parser.add_argument("--limit", type=int, default=20, help="Số bình luận tối đa lấy mỗi bài (mặc định 20)")
	parser.add_argument("--max-posts", type=int, default=None, help="Giới hạn số lượng bài viết cần cào")
	parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_CSV, help="Đường dẫn file CSV xuất ra")
	args = parser.parse_args()

	crawl_facebook_comments(
		input_file=args.input,
		output_file=args.output,
		batch_size=args.batch_size,
		limit_per_post=args.limit,
		max_posts=args.max_posts
	)