"""
Image-based background material service.

Provides search & download functions for multiple image providers:
  - Wikimedia Commons (free, no key)
  - Openverse (free, no key, CC-licensed)
  - Flickr (requires API key)
  - Google Custom Search (requires API key, 100 free/day)
  - Same Energy (experimental keyword search)
"""

import os
import random
import time
from typing import List, Optional
from urllib.parse import urlencode, quote_plus

import requests
from loguru import logger

from app.config import config
from app.models.schema import VideoAspect
from app.utils import utils

# ---------------------------------------------------------------------------
# Provider: Wikimedia Commons
# ---------------------------------------------------------------------------

def search_images_wikimedia(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[dict]:
    """Search Wikimedia Commons for images. No API key required."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {search_term}",
        "gsrlimit": per_page,
        "gsrnamespace": "6",  # File namespace
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": 1920,
    }
    url = f"https://commons.wikimedia.org/w/api.php?{urlencode(params)}"
    logger.info(f"[wikimedia] searching images: {search_term}")

    try:
        r = requests.get(url, timeout=(15, 30))
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        results = []
        for page in pages.values():
            ii = page.get("imageinfo", [{}])[0]
            mime = ii.get("mime", "")
            if not mime.startswith("image/"):
                continue
            img_url = ii.get("thumburl") or ii.get("url", "")
            if img_url:
                results.append({
                    "url": img_url,
                    "width": ii.get("thumbwidth", ii.get("width", 0)),
                    "height": ii.get("thumbheight", ii.get("height", 0)),
                    "provider": "wikimedia",
                    "search_term": search_term,
                })
        logger.info(f"[wikimedia] found {len(results)} images for '{search_term}'")
        return results
    except Exception as e:
        logger.error(f"[wikimedia] search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: Openverse
# ---------------------------------------------------------------------------

def search_images_openverse(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[dict]:
    """Search Openverse for CC-licensed images. No API key required."""
    aspect_ratio = "wide" if video_aspect == VideoAspect.landscape else "tall"
    params = {
        "q": search_term,
        "page_size": per_page,
        "aspect_ratio": aspect_ratio,
    }
    url = f"https://api.openverse.org/v1/images/?{urlencode(params)}"
    headers = {"User-Agent": "MoneyPrinterTurbo/1.0 (image background provider)"}
    logger.info(f"[openverse] searching images: {search_term}")

    try:
        r = requests.get(url, headers=headers, timeout=(15, 30))
        data = r.json()
        results = []
        for item in data.get("results", []):
            img_url = item.get("url", "")
            if img_url:
                results.append({
                    "url": img_url,
                    "width": item.get("width", 0),
                    "height": item.get("height", 0),
                    "provider": "openverse",
                    "search_term": search_term,
                })
        logger.info(f"[openverse] found {len(results)} images for '{search_term}'")
        return results
    except Exception as e:
        logger.error(f"[openverse] search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: Flickr
# ---------------------------------------------------------------------------

def search_images_flickr(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[dict]:
    """Search Flickr for images. Requires FLICKR_API_KEY."""
    api_key = config.flickr.get("api_key", "") or os.environ.get("FLICKR_API_KEY", "")
    if not api_key:
        logger.warning("[flickr] no API key configured, skipping")
        return []

    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "text": search_term,
        "sort": "relevance",
        "per_page": per_page,
        "format": "json",
        "nojsoncallback": 1,
        "content_type": 1,  # photos only
        "media": "photos",
        "extras": "url_l,url_o,url_c,url_z",  # large, original, medium-800, medium-640
        "license": "1,2,3,4,5,6,9,10",  # open licenses
    }
    url = f"https://api.flickr.com/services/rest/?{urlencode(params)}"
    logger.info(f"[flickr] searching images: {search_term}")

    try:
        r = requests.get(url, timeout=(15, 30))
        data = r.json()
        photos = data.get("photos", {}).get("photo", [])
        results = []
        for p in photos:
            # Prefer largest available URL
            img_url = p.get("url_l") or p.get("url_c") or p.get("url_o") or p.get("url_z", "")
            if img_url:
                w = int(p.get("width_l", 0) or p.get("width_c", 0) or p.get("width_o", 0) or 0)
                h = int(p.get("height_l", 0) or p.get("height_c", 0) or p.get("height_o", 0) or 0)
                results.append({
                    "url": img_url,
                    "width": w,
                    "height": h,
                    "provider": "flickr",
                    "search_term": search_term,
                })
        logger.info(f"[flickr] found {len(results)} images for '{search_term}'")
        return results
    except Exception as e:
        logger.error(f"[flickr] search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: Google Custom Search
# ---------------------------------------------------------------------------

def search_images_google(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 10,
) -> List[dict]:
    """Search Google Custom Search for images. 100 free queries/day."""
    api_key = config.google_cse.get("api_key", "") or os.environ.get("GOOGLE_CSE_API_KEY", "")
    cx = config.google_cse.get("cx", "") or os.environ.get("GOOGLE_CSE_CX", "")
    if not api_key or not cx:
        logger.warning("[google] no API key or CX configured, skipping")
        return []

    aspect = "wide" if video_aspect == VideoAspect.landscape else "tall"
    params = {
        "key": api_key,
        "cx": cx,
        "q": search_term,
        "searchType": "image",
        "num": min(per_page, 10),  # max 10 per request
        "imgSize": "xlarge",
        "imgType": "photo",
        "safe": "active",
    }
    if aspect == "wide":
        params["imgDominantColor"] = ""  # not used but keeps structure
    url = f"https://www.googleapis.com/customsearch/v1?{urlencode(params)}"
    logger.info(f"[google] searching images: {search_term}")

    try:
        r = requests.get(url, timeout=(15, 30))
        data = r.json()
        results = []
        for item in data.get("items", []):
            img = item.get("image", {})
            img_url = item.get("link", "")
            if img_url:
                results.append({
                    "url": img_url,
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                    "provider": "google",
                    "search_term": search_term,
                })
        logger.info(f"[google] found {len(results)} images for '{search_term}'")
        return results
    except Exception as e:
        logger.error(f"[google] search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Provider: Same Energy (experimental keyword search)
# ---------------------------------------------------------------------------

def search_images_same_energy(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[dict]:
    """Search Same Energy for visually similar images.
       Falls back to Openverse if unavailable.
    """
    logger.info(f"[same_energy] searching images: {search_term}")
    try:
        url = "https://api.same.energy/search"
        payload = {"query": search_term, "model": "v1", "num": per_page}
        headers = {"Content-Type": "application/json", "User-Agent": "MoneyPrinterTurbo/1.0"}
        r = requests.post(url, json=payload, headers=headers, timeout=(15, 30))
        if r.status_code != 200:
            raise Exception(f"HTTP {r.status_code}")
        data = r.json()
        results = []
        for item in data.get("results", []):
            img_url = item.get("image_url") or item.get("url", "")
            if img_url:
                results.append({
                    "url": img_url,
                    "width": item.get("width", 0),
                    "height": item.get("height", 0),
                    "provider": "same_energy",
                    "search_term": search_term,
                })
        logger.info(f"[same_energy] found {len(results)} images for '{search_term}'")
        return results
    except Exception as e:
        logger.warning(f"[same_energy] search failed ({e}), falling back to openverse")
        return search_images_openverse(search_term, video_aspect, per_page)


# ---------------------------------------------------------------------------
# Image download & save
# ---------------------------------------------------------------------------

def save_image(
    image_url: str,
    save_dir: str = "",
    search_term: str = "",
) -> str:
    """Download and save an image, returning its local path."""
    if not save_dir:
        save_dir = utils.storage_dir("cache_images", create=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url_hash = utils.md5(image_url.split("?")[0])
    # Determine extension
    ext = ".jpg"
    for check_ext in [".png", ".webp", ".jpeg", ".gif"]:
        if check_ext in image_url.lower():
            ext = check_ext
            break
    image_path = os.path.join(save_dir, f"img-{url_hash}{ext}")

    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        logger.debug(f"image already cached: {image_path}")
        return image_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        r = requests.get(
            image_url, headers=headers, proxies=config.proxy,
            verify=False, timeout=(30, 120), stream=True,
        )
        r.raise_for_status()
        with open(image_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        if os.path.exists(image_path) and os.path.getsize(image_path) > 1000:
            from PIL import Image as PILImage
            try:
                img = PILImage.open(image_path)
                img.verify()
                return image_path
            except Exception:
                os.remove(image_path)
                logger.warning(f"invalid image file: {image_path}")
                return ""
        else:
            if os.path.exists(image_path):
                os.remove(image_path)
            return ""
    except Exception as e:
        logger.error(f"failed to download image: {image_url} => {e}")
        if os.path.exists(image_path):
            os.remove(image_path)
        return ""


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

_SEARCH_FUNCS = {
    "wikimedia": search_images_wikimedia,
    "openverse": search_images_openverse,
    "flickr": search_images_flickr,
    "google": search_images_google,
    "same_energy": search_images_same_energy,
}

# Fallback order when the primary provider fails
_FALLBACK_ORDER = ["wikimedia", "openverse"]


def download_images(
    task_id: str,
    search_terms: List[str],
    provider: str = "wikimedia",
    video_aspect: VideoAspect = VideoAspect.portrait,
    audio_duration: float = 0.0,
    clip_duration: int = 5,
) -> List[str]:
    """Search and download images from the specified provider.
    
    Returns a list of local image file paths.
    Falls back through the fallback chain if the primary provider fails.
    """
    search_func = _SEARCH_FUNCS.get(provider)
    if not search_func:
        logger.error(f"unknown image provider: {provider}, falling back to wikimedia")
        search_func = search_images_wikimedia

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""
    if not material_directory:
        material_directory = utils.storage_dir("cache_images", create=True)

    all_image_items = []
    global_urls = set()

    for search_term in search_terms:
        items = search_func(search_term, video_aspect)

        # If primary provider returned nothing, try fallback
        if not items:
            for fb_provider in _FALLBACK_ORDER:
                if fb_provider == provider:
                    continue
                logger.info(f"primary provider '{provider}' returned 0 results, trying '{fb_provider}'")
                fb_func = _SEARCH_FUNCS[fb_provider]
                items = fb_func(search_term, video_aspect)
                if items:
                    break

        for item in items:
            if item["url"] not in global_urls:
                global_urls.add(item["url"])
                all_image_items.append(item)

    logger.info(f"found {len(all_image_items)} unique images across {len(search_terms)} search terms")

    # Calculate how many images we need
    needed_images = max(1, int(audio_duration / clip_duration) + 1)
    logger.info(f"need approximately {needed_images} images for {audio_duration:.1f}s audio ({clip_duration}s per image)")

    # Shuffle for variety
    random.shuffle(all_image_items)

    image_paths = []
    for item in all_image_items[:needed_images * 2]:  # fetch extra for redundancy
        saved_path = save_image(
            image_url=item["url"],
            save_dir=material_directory,
            search_term=item.get("search_term", ""),
        )
        if saved_path:
            image_paths.append(saved_path)
            if len(image_paths) >= needed_images:
                break

    logger.success(f"downloaded {len(image_paths)} images for task {task_id}")
    return image_paths
