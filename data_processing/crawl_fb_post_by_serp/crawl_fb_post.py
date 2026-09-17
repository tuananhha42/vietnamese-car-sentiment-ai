# -*- coding: utf-8 -*-
import argparse
import csv
import datetime
import logging
import os
import sys
import time
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
import requests
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSING_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(DATA_PROCESSING_DIR)

# Tự động nạp cấu hình từ .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY_FILE = os.path.join(CURRENT_DIR, "serp_api_key.txt")
LOG_FILE = os.path.join(CURRENT_DIR, "log.txt")
DEFAULT_OUTPUT_FILE = os.path.join(CURRENT_DIR, "facebook_car_posts.csv")
SERPAPI_URL = "https://serpapi.com/search.json"

FIELDNAMES = ["url", "post_type", "title", "snippet", "query", "crawled_at"]

# Cấu hình logger
logger = logging.getLogger("crawl_fb_post")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Handler xuất ra terminal để theo dõi tiến trình
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Handler chỉ lưu log lỗi vào file log.txt trong cùng thư mục
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.ERROR)
    file_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

# Bộ từ khóa tối ưu: Đơn giản, rõ ràng, không lồng ngoặc phức tạp để Google trả nhiều kết quả nhất
MODELS = [
    "VinFast VF3", "VinFast VF5", "VinFast VF8", "VinFast VF7", "VinFast VF9",
    "Mazda CX-5", "Hyundai Santa Fe", "Hyundai Tucson", "Honda CR-V", "Kia Seltos",
    "Toyota Vios", "Hyundai Accent", "Honda City", "Kia K3", "Mazda 3",
    "Mitsubishi Xpander", "Toyota Veloz", "Kia Carnival", "Ford Everest", "Ford Ranger",
    "Toyota Corolla Cross", "Hyundai Creta", "Kia Sonet", "Toyota Fortuner"
]

TOPICS = [
    "đánh giá ưu nhược điểm",
    "trải nghiệm thực tế",
    "lỗi thường gặp",
    "chi phí nuôi xe bảo dưỡng",
    "khen chê chia sẻ kinh nghiệm"
]

# Tự động sinh danh sách query phủ rộng
SEARCH_QUERIES = [f'site:facebook.com "{model}" "{topic}"' for model in MODELS for topic in TOPICS]
# Bổ sung các query cộng đồng/hội nhóm
SEARCH_QUERIES.extend([
    'site:facebook.com/groups "hội review xe ô tô" chia sẻ',
    'site:facebook.com/groups "kinh nghiệm lái xe ô tô" hỏi đáp',
    'site:facebook.com "tư vấn mua xe ô tô" tầm giá',
    'site:facebook.com "đánh giá xe ô tô sau 1 năm sử dụng"'
])


def load_api_keys(file_path=API_KEY_FILE):
    # 1. Ưu tiên đọc từ biến môi trường SERP_API_KEYS (.env)
    env_keys = os.getenv("SERP_API_KEYS", "").strip()
    if env_keys:
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        if keys:
            logger.info(f"[*] Đã nạp thành công {len(keys)} SerpApi keys từ biến môi trường (.env)")
            return keys

    # 2. Đọc dự phòng từ file text nếu có
    if os.path.exists(file_path):
        logger.warning(f"[!] Không tìm thấy SERP_API_KEYS trong .env. Đang đọc dự phòng từ: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            keys = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if keys:
            return keys

    err = "Không tìm thấy SerpApi key hợp lệ trong biến môi trường SERP_API_KEYS (.env) hoặc file text!"
    logger.error(err)
    raise ValueError(err)


class SerpApiManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_key_idx = 0

    def get_current_key(self):
        if self.current_key_idx >= len(self.keys):
            return None
        return self.keys[self.current_key_idx]

    def switch_next_key(self):
        self.current_key_idx += 1
        if self.current_key_idx < len(self.keys):
            logger.info(f"[+] Chuyển sang API Key [{self.current_key_idx + 1}/{len(self.keys)}]")
            return True
        logger.error("Đã dùng hết danh sách API Keys!")
        return False


def is_valid_facebook_post_url(url):
    if not url or "facebook.com" not in url:
        return False
    path = urlparse(url).path.lower()
    valid_indicators = [
        "/posts/", "/permalink/", "/permalink.php", "/story.php",
        "/videos/", "/watch", "/reel/", "/reels/", "/photo.php", "/photo/"
    ]
    if not any(ind in path for ind in valid_indicators):
        if "story_fbid=" not in url and "fbid=" not in url:
            return False

    blacklist = ["/search", "/login", "/recover", "/help", "/policies", "/directory"]
    return not any(bp in path for bp in blacklist)


def clean_facebook_url(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") if len(parsed.path) > 1 else parsed.path
    keep_params = {"story_fbid", "id", "fbid", "v"}
    query_dict = parse_qs(parsed.query)
    clean_query = {k: v[0] for k, v in query_dict.items() if k in keep_params}

    if clean_query:
        sorted_params = urlencode(sorted(clean_query.items()))
        return urlunparse((parsed.scheme or "https", "www.facebook.com", path, "", sorted_params, ""))
    return urlunparse((parsed.scheme or "https", "www.facebook.com", path, "", "", ""))


def detect_post_type(url):
    u = url.lower()
    if "/groups/" in u:
        return "group_post"
    if "/reel/" in u or "/reels/" in u:
        return "reel"
    if "/videos/" in u or "/watch" in u:
        return "video"
    if "/photo" in u:
        return "photo_post"
    return "page_or_user_post"


def append_rows_to_csv(file_path, new_rows):
    if not new_rows:
        return
    try:
        file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
        with open(file_path, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_rows)
    except Exception as e:
        logger.error(f"Lỗi ghi dữ liệu vào file CSV ({file_path}): {e}")


def scrape_facebook_urls(target_count=1000, output_file=None, delay=1.0, api_key_file=API_KEY_FILE):
    if output_file is None:
        output_file = DEFAULT_OUTPUT_FILE

    api_keys = load_api_keys(api_key_file)
    api_manager = SerpApiManager(api_keys)
    collected_urls = set()

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, mode="r", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if r.get("url"):
                        collected_urls.add(r["url"])
            logger.info(f"[*] Đã tải {len(collected_urls)} URLs đã có từ file.")
        except Exception as e:
            logger.error(f"Lỗi đọc file kết quả cũ ({output_file}): {e}")

    logger.info("=" * 70)
    logger.info(f"[*] BẮT ĐẦU CÀO URL MỤC TIÊU: {target_count}")
    logger.info(f"[*] Tổng số Query sẵn sàng: {len(SEARCH_QUERIES)}")
    logger.info("=" * 70)

    for q_idx, query in enumerate(SEARCH_QUERIES, 1):
        if len(collected_urls) >= target_count:
            break

        logger.info(f"\n[{q_idx}/{len(SEARCH_QUERIES)}] {query}")
        start_page = 0

        while len(collected_urls) < target_count and start_page < 40:  # Tối đa 2 trang / query để giữ độ tươi mới
            api_key = api_manager.get_current_key()
            if not api_key:
                logger.error("Hết API key khả dụng để tiếp tục cào.")
                return output_file

            params = {
                "engine": "google",
                "q": query,
                "hl": "vi",
                "gl": "vn",
                "num": 20,
                "start": start_page,
                "api_key": api_key
            }

            response_data = None
            last_error = None
            for attempt in range(3):  # Thử lại 3 lần nếu mạng timeout
                try:
                    res = requests.get(SERPAPI_URL, params=params, timeout=60)
                    if res.status_code == 429 or (res.status_code == 400 and "quota" in res.text.lower()):
                        logger.error(f"[-] Key [{api_manager.current_key_idx + 1}] hết quota hoặc bị rate limit (HTTP {res.status_code}) tại query: '{query}'. Đang chuyển key...")
                        api_manager.switch_next_key()
                        params["api_key"] = api_manager.get_current_key()
                        continue
                    if res.status_code == 200:
                        response_data = res.json()
                        break
                    err_msg = f"HTTP {res.status_code} tại query '{query}' trang {start_page//20 + 1}: {res.text[:150]}"
                    logger.warning(f"[!] {err_msg}, thử lại lần {attempt+1}...")
                    last_error = err_msg
                except requests.exceptions.RequestException as e:
                    err_msg = f"Timeout/Lỗi mạng tại query '{query}' trang {start_page//20 + 1}: {e}"
                    logger.warning(f"[-] {err_msg}. Thử lại {attempt+1}/3...")
                    last_error = err_msg
                    time.sleep(2)
                except Exception as e:
                    err_msg = f"Lỗi không xác định khi gọi SerpAPI: {e}"
                    logger.error(f"[!] {err_msg}")
                    last_error = err_msg
                    break

            if not response_data:
                if last_error:
                    logger.error(f"Thất bại sau 3 lần thử lại: {last_error}")
                break

            results = response_data.get("organic_results", [])
            if not results:
                break

            batch_rows = []
            for item in results:
                raw_link = item.get("link", "")
                if is_valid_facebook_post_url(raw_link):
                    clean_url = clean_facebook_url(raw_link)
                    if clean_url not in collected_urls:
                        collected_urls.add(clean_url)
                        batch_rows.append({
                            "url": clean_url,
                            "post_type": detect_post_type(clean_url),
                            "title": item.get("title", "").strip(),
                            "snippet": item.get("snippet", "").strip(),
                            "query": query,
                            "crawled_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

            if batch_rows:
                append_rows_to_csv(output_file, batch_rows)

            logger.info(f"  -> +{len(batch_rows)} link mới | Tổng: {len(collected_urls)}/{target_count}")

            if len(collected_urls) >= target_count or "next" not in response_data.get("pagination", {}):
                break

            start_page += 20
            time.sleep(delay)

    logger.info(f"\n[✔] HOÀN THÀNH. Tổng URLs: {len(collected_urls)}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--api-key-file", type=str, default=API_KEY_FILE)
    args = parser.parse_args()

    scrape_facebook_urls(target_count=args.target, output_file=args.output, delay=args.delay, api_key_file=args.api_key_file)