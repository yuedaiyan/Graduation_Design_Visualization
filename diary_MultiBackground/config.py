"""User-tunable parameters and background source definitions."""

from __future__ import annotations

import os


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


BASE_CONTENT_SIZE = 1000
DEFAULT_CONTENT_SIZE = 1000
DEFAULT_RESOLUTION = 2160

CANVAS_SIZE = DEFAULT_CONTENT_SIZE
OUTPUT_SIZE = DEFAULT_RESOLUTION
DPI = 200
RANDOM_SEED = 42
OUTPUT_IMAGE_FORMATS = env_list("MULTIBG_OUTPUT_IMAGE_FORMATS", ["svg", "png"])

# Use the HuggingFace mirror in China by default. Run with HF_ENDPOINT="" if the
# mirror is unavailable and you want the official HuggingFace host.
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")

# Which backgrounds to build. Use ["all"] to run every configured background.
# This expanded Chinese set keeps English Wikipedia disabled, and adds the
# discourse families that are closer to the diary project.
ENABLED_BACKGROUNDS = [
    "zh_wikipedia",
    "thucnews",
    "zhihu_kol",
    "douban_reviews",
    "weibo_senti",
    "classical_poetry",
    "historical_diary",
    "modern_essay",
    "psych_discourse",
    "legal_text",
    "religious_text",
    "lyrics",
    "interview_oral",
    "forum_emotional",
    "self_help",
    "travel_writing",
    "food_writing",
    "academic_humanities",
    "children_writing",
    "ad_copy",
    "dream_record",
]
ENABLED_BACKGROUNDS = env_list("MULTIBG_ENABLED_BACKGROUNDS", ENABLED_BACKGROUNDS)

# Download / sample cache size. This controls the size of saved plain text cache,
# not the final embedding size. For a tiny first run, try 5 * 1024 * 1024.
TEXT_CACHE_MAX_BYTES = 200 * 1024 * 1024
TEXT_CACHE_MAX_BYTES = env_int("MULTIBG_TEXT_CACHE_MAX_BYTES", TEXT_CACHE_MAX_BYTES)

# Stop collecting once either target is reached.
TEXT_CACHE_TARGET_ITEMS = 40000
TEXT_CACHE_MIN_ITEMS = 500
TEXT_CACHE_TARGET_ITEMS = env_int("MULTIBG_TEXT_CACHE_TARGET_ITEMS", TEXT_CACHE_TARGET_ITEMS)
TEXT_CACHE_MIN_ITEMS = env_int("MULTIBG_TEXT_CACHE_MIN_ITEMS", TEXT_CACHE_MIN_ITEMS)

# Actual number of cached texts used for embedding + PCA/UMAP background.
# For a tiny first run, try 800-2000. For final images, try 20000-30000.
VECTORIZE_TEXT_LIMIT = 7000
VECTORIZE_TEXT_LIMIT = env_int("MULTIBG_VECTORIZE_TEXT_LIMIT", VECTORIZE_TEXT_LIMIT)

# Per-source filtering before chunking.
MIN_TEXT_CHARS = 120
TITLE_PREFIX = True

# Background points should be comparable with one diary day. Long source texts are
# split into chunks; naturally short texts are packed together before embedding.
TEXT_CACHE_SCHEMA_VERSION = "chunked_stratified_v1"
BACKGROUND_CHUNK_MIN_CHARS = 300
BACKGROUND_CHUNK_TARGET_CHARS = 420
BACKGROUND_CHUNK_MAX_CHARS = 500
PACK_SHORT_TEXTS = True

# Cap any one sampling stratum so frequent source types, such as Wikipedia
# places or biographies, cannot dominate the background.
STRATIFIED_SAMPLING = True
STRATUM_MAX_SHARE = 0.22
STRATUM_MAX_SHARE = env_float("MULTIBG_STRATUM_MAX_SHARE", STRATUM_MAX_SHARE)
STRATUM_EXPECTED_COUNTS = {
    "zh_wikipedia": 8,
    "weibo_senti": 2,
    # These datasets often expose weak or inconsistent category fields. Keep
    # their strata in the cache for audit, but do not under-fill the background
    # just because the public dataset lacks reliable labels.
    "thucnews": 1,
    "zhihu_kol": 1,
    "douban_reviews": 1,
    "classical_poetry": 1,
}

# Streaming shuffle is an approximate random sample across the stream.
# Higher is more representative but may read more memory/network.
SHUFFLE_BUFFER_SIZE = 120000
SHUFFLE_BUFFER_SIZE = env_int("MULTIBG_SHUFFLE_BUFFER_SIZE", SHUFFLE_BUFFER_SIZE)
MAX_SCAN_ROWS_OVERRIDE = env_int("MULTIBG_MAX_SCAN_ROWS", 0)

DIARY_VECTORS_DIR = "../diary_vectors"
DIARY_TEXT_JSON = os.environ.get("MULTIBG_DIARY_TEXT_JSON", "../diary_entries.merged.json")
LOCAL_MODEL_DIR = "../Qwen3-Embedding-0.6B"
OUTPUT_ROOT = "../output_All/diary_MultiBackground"
CACHE_DIR = ".cache"
LOCAL_CORPUS_ROOT = "corpora"

EMBED_MAX_TOKENS = 256
EMBED_BATCH_SIZE = 32
EMBED_PART_SIZE = 512
EMBED_SORT_BY_LENGTH = True

REDUCER = "pca"  # "pca" for global area, "umap" for local neighborhoods.
FIT_REDUCER_ON = "background"  # "background" or "combined".
UMAP_N_NEIGHBORS = 45
UMAP_MIN_DIST = 0.12
PCA_WHITEN = False

# Convex-hull boundaries are computed from the central share of each point set,
# so a few far outliers do not define the visual/statistical area.
HULL_KEEP_RATIO = env_float("MULTIBG_HULL_KEEP_RATIO", 0.95)
DIARY_AREA_SCALE_RATIO = env_float("MULTIBG_DIARY_AREA_SCALE_RATIO", 0.50)

FIGURE_MARGIN_RATIO = env_float("MULTIBG_FIGURE_MARGIN_RATIO", 0.045)
FIGURE_TITLE_TOP = env_float("MULTIBG_FIGURE_TITLE_TOP", 0.90)
WORLD_PAD_RATIO = env_float("MULTIBG_WORLD_PAD_RATIO", 0.09)
ZOOM_PAD_RATIO = env_float("MULTIBG_ZOOM_PAD_RATIO", 0.28)
ZOOM_EDGE_INSET_RATIO = env_float("MULTIBG_ZOOM_EDGE_INSET_RATIO", 0.018)

BACKGROUND_POINT_SIZE = 1.15
DIARY_POINT_SIZE = 1.85
BACKGROUND_ZOOM_POINT_SIZE = 1.15
DIARY_ZOOM_POINT_SIZE = 2.15
BACKGROUND_ALPHA = 1.0
DIARY_ALPHA = 0.96
SHOW_DIARY_LABEL_EVERY = 0  # Date labels are only visual indexes; 0 disables them.

BACKGROUND_COLOR = "none"
WORLD_COLOR = "#6f7275"
DIARY_COLOR = "#242426"
DIARY_HULL_COLOR = "#aa1116"
DIARY_HULL_LINEWIDTH = 0.25
TEXT_COLOR = "#222222"
TITLE_FONT_FAMILY = "Akzidenz-Grotesk BQ"
TITLE_FONT_SIZE_PT = 8
TITLE_LINE_HEIGHT_PT = 11

THEME_RGB_COLORS = [
    (246, 169, 174),
    (238, 135, 145),
    (226, 103, 118),
    (207, 78, 96),
    (248, 187, 122),
    (239, 158, 91),
    (224, 132, 72),
    (202, 106, 58),
    (244, 215, 118),
    (230, 195, 87),
    (211, 172, 63),
    (188, 146, 48),
    (166, 209, 139),
    (135, 189, 118),
    (105, 164, 98),
    (79, 139, 83),
    (132, 204, 196),
    (101, 181, 180),
    (77, 154, 166),
    (62, 127, 148),
    (143, 173, 222),
    (121, 144, 208),
    (105, 116, 190),
    (91, 91, 164),
    (191, 183, 207),
    (164, 153, 187),
    (137, 124, 164),
    (113, 99, 141),
]

# Supported source kinds:
# - "hf_dataset": datasets.load_dataset(name, config, split, streaming=True)
# - "hf_parquet": datasets.load_dataset("parquet", data_files=[...], streaming=True)
# - "hf_search_dataset": same as hf_dataset, but kept separate in names to mark
#   manually selected datasets from a search page.
# - "local_texts": read .txt/.md/.json/.jsonl/.csv files from local paths.
# - "url_json": read JSON records from one or more direct URLs.
# - "mixed_sources": try several sources in order; failed/empty sources are skipped.
#
# Field extraction is intentionally flexible because these public datasets use
# different schemas. The first usable text-like field wins unless text_fields is set.
BACKGROUND_SPECS = {
    "zh_wikipedia": {
        "title": "Chinese knowledge: Chinese Wikipedia",
        "source_kind": "hf_dataset",
        "dataset_name": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "text_fields": ["text"],
        "title_fields": ["title"],
        "min_text_chars": 180,
        "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
    },
    "thucnews": {
        "title": "Chinese contemporary media: THUCNews-like titles",
        "source_kind": "local_texts",
        "paths": [f"{LOCAL_CORPUS_ROOT}/thucnews"],
        "text_fields": ["content", "title", "text", "sentence"],
        "title_fields": ["label", "category"],
        "min_text_chars": 20,
    },
    "zhihu_kol": {
        "title": "Private conversational writing: Zhihu Q&A",
        "source_kind": "hf_dataset",
        "dataset_name": "wangrui6/Zhihu-KOL",
        "config": None,
        "split": "train",
        "text_fields": [
            "answer",
            "content",
            "text",
            "question",
            "title",
            "description",
        ],
        "title_fields": ["question", "title"],
        "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
    },
    "douban_reviews": {
        "title": "Private feeling writing: Douban movie reviews",
        "source_kind": "hf_dataset",
        "dataset_name": "GT610/douban",
        "config": None,
        "split": "train",
        "text_fields": ["text", "review", "comment", "content", "short_comment"],
        "title_fields": ["movie", "title", "name"],
        "min_text_chars": 8,
        "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
    },
    "weibo_senti": {
        "title": "Colloquial private writing: Weibo sentiment",
        "source_kind": "local_texts",
        "paths": [f"{LOCAL_CORPUS_ROOT}/weibo_senti"],
        "text_fields": ["review", "text", "content", "sentence"],
        "title_fields": ["label"],
        "min_text_chars": 4,
    },
    "classical_poetry": {
        "title": "Literary tradition: classical Chinese poetry",
        "source_kind": "hf_dataset",
        "dataset_name": "Ayaka/ORCHESTRA-simple-1M",
        "config": None,
        "split": "train",
        "text_fields": ["text", "content", "poem", "paragraphs", "sentence"],
        "title_fields": ["title", "author", "dynasty"],
        "min_text_chars": 8,
        "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
    },
    "historical_diary": {
        "title": "Historical private writing: diaries and letters",
        "source_kind": "mixed_sources",
        "text_fields": ["text", "content", "body", "paragraphs"],
        "title_fields": ["title", "name", "author"],
        "min_text_chars": 80,
        "sources": [
            {
                "source_name": "local curated historical diary and letters",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/historical_diary"],
            },
            {
                "source_name": "Chinese Wikisource filtered diary and letters",
                "source_kind": "hf_dataset",
                "dataset_name": "wikimedia/wikisource",
                "config": "20231201.zh",
                "split": "train",
                "include_keywords": ["日记", "書信", "书信", "家书", "尺牍", "信札", "札记", "游记"],
                "filter_fields": ["title", "text"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
    "modern_essay": {
        "title": "Modern literary prose: essays and reflective writing",
        "source_kind": "mixed_sources",
        "text_fields": ["text", "content", "body", "paragraphs"],
        "title_fields": ["title", "name", "author"],
        "min_text_chars": 120,
        "sources": [
            {
                "source_name": "local curated modern essays",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/modern_essay"],
            },
            {
                "source_name": "Chinese Wikisource modern prose filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wikimedia/wikisource",
                "config": "20231201.zh",
                "split": "train",
                "include_keywords": ["散文", "随笔", "杂文", "鲁迅", "周作人", "朱自清", "冰心", "林语堂"],
                "filter_fields": ["title", "text"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
    "psych_discourse": {
        "title": "Psychological discourse: counseling and self-description",
        "source_kind": "mixed_sources",
        "text_fields": ["answer", "response", "content", "text", "question", "dialogue", "conversation"],
        "title_fields": ["question", "title", "topic", "label"],
        "min_text_chars": 80,
        "sources": [
            {
                "source_name": "local curated psychology discourse",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/psych_discourse"],
            },
            {
                "source_name": "PsyDial-D2 Chinese psychological counseling",
                "source_kind": "hf_dataset",
                "dataset_name": "qiuhuachuan/PsyDial-D2",
                "config": None,
                "split": "train",
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
            },
            {
                "source_name": "Chinese Wikipedia psychology filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wikimedia/wikipedia",
                "config": "20231101.zh",
                "split": "train",
                "text_fields": ["text"],
                "title_fields": ["title"],
                "include_keywords": ["心理", "咨询", "焦虑", "抑郁", "创伤", "治疗", "人格", "情绪"],
                "filter_fields": ["title", "text"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 400000,
            },
        ],
    },
    "legal_text": {
        "title": "Legal and policy language: public institutional writing",
        "source_kind": "mixed_sources",
        "text_fields": ["fact", "text", "content", "case", "document", "body"],
        "title_fields": ["charge", "accusation", "title", "category"],
        "min_text_chars": 120,
        "sources": [
            {
                "source_name": "local curated legal and policy text",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/legal_text"],
            },
            {
                "source_name": "CAIL2018 Chinese legal cases",
                "source_kind": "hf_dataset",
                "dataset_name": "china-ai-law-challenge/cail2018",
                "config": None,
                "split": "first_stage_train",
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
            },
        ],
    },
    "religious_text": {
        "title": "Religious and spiritual text: older inner techniques",
        "source_kind": "mixed_sources",
        "text_fields": ["text", "content", "body", "paragraphs"],
        "title_fields": ["title", "name", "author"],
        "min_text_chars": 80,
        "sources": [
            {
                "source_name": "local curated religious text",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/religious_text"],
            },
            {
                "source_name": "Chinese Wikisource religious filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wikimedia/wikisource",
                "config": "20231201.zh",
                "split": "train",
                "include_keywords": ["佛", "经", "聖經", "圣经", "道德经", "庄子", "心经", "金刚经", "法华经", "坛经"],
                "filter_fields": ["title", "text"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
    "lyrics": {
        "title": "Designed private feeling: Chinese popular lyrics",
        "source_kind": "mixed_sources",
        "text_fields": ["lyric", "lyrics", "text", "content"],
        "title_fields": ["song", "title", "artist", "name"],
        "min_text_chars": 8,
        "sources": [
            {
                "source_name": "local curated lyrics",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/lyrics"],
            },
            {
                "source_name": "ChineseLyrics GitHub lyrics shard",
                "source_kind": "url_json",
                "urls": [
                    "https://raw.githubusercontent.com/dengxiuqi/ChineseLyrics/master/lyrics1.json"
                ],
            },
        ],
    },
    "interview_oral": {
        "title": "Interview and oral narration: structured spoken self-report",
        "source_kind": "mixed_sources",
        "text_fields": ["answer", "response", "content", "text", "dialogue", "conversation"],
        "title_fields": ["question", "title", "speaker", "role"],
        "min_text_chars": 60,
        "sources": [
            {
                "source_name": "local curated interview and oral history",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/interview_oral"],
            },
            {
                "source_name": "Zhihu interview-like Q&A filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wangrui6/Zhihu-KOL",
                "config": None,
                "split": "train",
                "include_keywords": ["采访", "访谈", "口述", "记者", "问：", "答：", "我的故事"],
                "filter_fields": ["question", "title", "answer", "content"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
    "forum_emotional": {
        "title": "Anonymous emotional forum writing: long-form confessional text",
        "source_kind": "mixed_sources",
        "text_fields": ["answer", "content", "text", "review", "question"],
        "title_fields": ["question", "title", "topic", "label"],
        "min_text_chars": 60,
        "sources": [
            {
                "source_name": "local curated emotional forum and treehole text",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/forum_emotional"],
            },
            {
                "source_name": "Zhihu emotional anonymous-style filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wangrui6/Zhihu-KOL",
                "config": None,
                "split": "train",
                "include_keywords": ["匿名", "树洞", "倾诉", "分手", "焦虑", "抑郁", "崩溃", "难过", "孤独"],
                "filter_fields": ["question", "title", "answer", "content"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
    "self_help": {
        "title": "Self-help and motivational writing: public discipline in private voice",
        "source_kind": "mixed_sources",
        "text_fields": ["text", "content", "body", "review", "answer"],
        "title_fields": ["title", "topic", "category", "label"],
        "min_text_chars": 80,
        "sources": [
            {
                "source_name": "local curated self-help writing",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/self_help"],
            },
            {
                "source_name": "Zhihu self-help filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wangrui6/Zhihu-KOL",
                "config": None,
                "split": "train",
                "include_keywords": ["成长", "自律", "成功", "励志", "人生", "复盘", "认知", "改变自己", "个人成长"],
                "filter_fields": ["question", "title", "answer", "content"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
    "travel_writing": {
        "title": "Travel and place writing: public spatial narration",
        "source_kind": "mixed_sources",
        "text_fields": ["text", "content", "body", "review"],
        "title_fields": ["title", "place", "destination", "category"],
        "min_text_chars": 80,
        "sources": [
            {
                "source_name": "local curated travel writing",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/travel_writing"],
            },
            {
                "source_name": "Zhihu travel writing filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wangrui6/Zhihu-KOL",
                "config": None,
                "split": "train",
                "include_keywords": ["旅行", "旅游", "游记", "攻略", "景点", "酒店", "民宿", "城市漫步", "出发"],
                "filter_fields": ["question", "title", "answer", "content"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
    "food_writing": {
        "title": "Food and lifestyle writing: public daily materiality",
        "source_kind": "mixed_sources",
        "text_fields": ["text", "content", "body", "review", "comment"],
        "title_fields": ["title", "dish", "name", "category"],
        "min_text_chars": 20,
        "sources": [
            {
                "source_name": "local curated food writing",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/food_writing"],
            },
            {
                "source_name": "Zhihu food and lifestyle filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wangrui6/Zhihu-KOL",
                "config": None,
                "split": "train",
                "include_keywords": ["美食", "菜谱", "做法", "食材", "餐厅", "下厨房", "烘焙", "早餐", "晚餐"],
                "filter_fields": ["question", "title", "answer", "content"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
    "academic_humanities": {
        "title": "Humanities academic discourse: meta-language of self and writing",
        "source_kind": "mixed_sources",
        "text_fields": ["abstract", "text", "content", "body"],
        "title_fields": ["title", "subject", "category", "label"],
        "min_text_chars": 120,
        "sources": [
            {
                "source_name": "local curated humanities abstracts",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/academic_humanities"],
            },
            {
                "source_name": "Chinese Wikipedia humanities filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wikimedia/wikipedia",
                "config": "20231101.zh",
                "split": "train",
                "text_fields": ["text"],
                "title_fields": ["title"],
                "include_keywords": ["哲学", "文学理论", "叙事", "主体", "自我", "日常生活", "现代性", "现象学", "阐释学"],
                "filter_fields": ["title", "text"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 400000,
            },
        ],
    },
    "children_writing": {
        "title": "Student composition: disciplined private writing",
        "source_kind": "mixed_sources",
        "text_fields": ["essay", "text", "content", "answer", "response", "prompt", "RESPONSE"],
        "title_fields": ["title", "prompt", "topic", "grade"],
        "min_text_chars": 60,
        "sources": [
            {
                "source_name": "local curated student compositions",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/children_writing"],
            },
            {
                "source_name": "Chinese writing benchmark",
                "source_kind": "hf_dataset",
                "dataset_name": "zake7749/chinese-writing-benchmark",
                "config": None,
                "split": "gpt_5.4",
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
            },
        ],
    },
    "ad_copy": {
        "title": "Advertising copy: designed outward language",
        "source_kind": "mixed_sources",
        "text_fields": ["text", "content", "copy", "body", "title"],
        "title_fields": ["title", "brand", "product", "category"],
        "min_text_chars": 20,
        "sources": [
            {
                "source_name": "local curated advertising copy",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/ad_copy"],
            },
            {
                "source_name": "Zhihu advertising and seeding-copy filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wangrui6/Zhihu-KOL",
                "config": None,
                "split": "train",
                "include_keywords": ["品牌", "广告", "文案", "种草", "优惠", "新品", "购买", "体验官", "限时"],
                "filter_fields": ["question", "title", "answer", "content"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
    "dream_record": {
        "title": "Dream records: extreme form of clear and obscure private writing",
        "source_kind": "mixed_sources",
        "text_fields": ["answer", "content", "text", "question", "body"],
        "title_fields": ["question", "title", "topic"],
        "min_text_chars": 40,
        "sources": [
            {
                "source_name": "local curated dream records",
                "source_kind": "local_texts",
                "paths": [f"{LOCAL_CORPUS_ROOT}/dream_record"],
            },
            {
                "source_name": "Zhihu dream-record filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wangrui6/Zhihu-KOL",
                "config": None,
                "split": "train",
                "include_keywords": ["梦见", "梦到", "做梦", "梦境", "噩梦", "清醒梦"],
                "filter_fields": ["question", "title", "answer", "content"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
            {
                "source_name": "Chinese Wikisource dream and psychoanalysis filter",
                "source_kind": "hf_dataset",
                "dataset_name": "wikimedia/wikisource",
                "config": "20231201.zh",
                "split": "train",
                "include_keywords": ["梦", "夢", "解梦", "释梦", "潜意识"],
                "filter_fields": ["title", "text"],
                "shuffle_buffer": SHUFFLE_BUFFER_SIZE,
                "max_scan_rows": 500000,
            },
        ],
    },
}
