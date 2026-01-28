# update base files
from ._FUNCS_ import *
from .MAP import convert_XY

__all__ = ['CO_BASEDATA',
           'Get_LAS',
           'Get_ProdData',
           'Get_Scouts',
           'Merge_Frac_Focus',
           'SUMMARIZE_COGCC',
           'SUMMARIZE_PROD_DATA', 
           'SUMMARIZE_PROD_DATA2',
           'CO_Get_Surveys',
           'SUMMARIZE_COGCC_SQL']

#Define Functions for multiprocessing iteration
# Def function to add metrics to three stream rates (TMB, Norm, Cum)

def ND_WELLSUMMARY(username, password, driver= None):
    if driver == None:
        driver = get_driver()

    # tops, tops info, well stuff(has NCIS number!)
    FLAT_FILE_URL = 'https://www.dmr.nd.gov/oilgas/feeservices/flatfiles/flatfiles.asp'

    driver = login_to_website(username, password, FLAT_FILE_URL, driver)

    soup = BS(driver.page_source)
    links = soup.find_all("a")

    FILES = []
    for f in links:
        ftxt = f.get('href')
        if 'flatfiles' in ftxt:
            DL_FILE = ftxt.split('/')[-1]
            FILES.append(DL_FILE)
            if path.exists(DL_FILE):
               remove(DL_FILE)
            e = driver.find_element(webdriver.common.by.By.LINK_TEXT,f.text)
            e.click()
    for f in FILES:
        with ZipFile(f, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall('ND_WELLDATA')
    driver.quit()

def filename_from_request(url):
    response = requests.get(url, stream=True)
    if "content-disposition" in response.headers:
        import re
        cd = response.headers["content-disposition"]
        match = re.findall('filename="?([^"]+)"?', cd)
        if match:
            filename = match[0]
        else:
            filename = "downloaded_file"
    else:
        from urllib.parse import urlparse
        import os
    filename = path.basename(urlparse(url).path)
    return filename

#def filename_from_content_disposition(headers: dict) -> (str | None):
def filename_from_content_disposition(headers: dict):
    """
    Parse RFC 6266/5987 Content-Disposition for filename/filename*.
    Returns None if unavailable.
    """
    cd = headers.get("Content-Disposition") or headers.get("content-disposition")
    if not cd:
        return None

    # filename*=UTF-8''encoded-name.ext (RFC 5987)
    m = re.search(r"filename\*\s*=\s*([^']*)''([^;]+)", cd, flags=re.IGNORECASE)
    if m:
        # charset = m.group(1)  # usually UTF-8
        return unquote(m.group(2))

    # filename="name.ext" OR filename=name.ext
    m = re.search(r'filename\s*=\s*"([^"]+)"', cd, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'filename\s*=\s*([^;]+)', cd, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def filename_from_url(url: str) -> str:
    upath = urlparse(url).path
    name = path.basename(upath.rstrip("/"))
    return name or "downloaded_file"


def safe_extract(zip_path: Path, dest_dir: Path) -> list[Path]:
    """
    Extracts ZIP while preventing zip-slip (path traversal).
    Returns list of extracted file paths (relative to dest_dir).
    """
    extracted = []
    dest_dir = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            # Skip directory entries cleanly
            if member.filename.endswith("/"):
                continue
            # Normalize and prevent absolute/parent paths
            target = (dest_dir / member.filename).resolve()
            if not str(target).startswith(str(dest_dir) + sep):
                # Attempted path traversal; skip
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted


def download_and_extract_zip(
    url: str,
    dest_dir: str,                # str | Path
    filename: str = None,         # str | None = None
    chunk_size: int = 1 << 20,  # 1 MiB
    timeout: int = 60,
) -> list[Path]:
    """
    Streams a ZIP from `url` to disk and extracts it to `dest_dir`.
    - If `filename` not provided, tries Content-Disposition then falls back to URL.
    - Returns list of extracted file Paths.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        # Decide filename
        cd_name = filename_from_content_disposition(r.headers)
        fname = filename or cd_name or filename_from_url(url)
        # Give it a .zip suffix if missing and server didn’t provide one
        if not fname.lower().endswith(".zip"):
            fname += ".zip"
        zip_path = dest_dir / fname

        # Stream to disk
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # skip keep-alive chunks
                    f.write(chunk)

    # Validate and extract
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Downloaded file is not a valid ZIP: {zip_path}")

    extracted = safe_extract(zip_path, dest_dir)
    return extracted

def CO_BASEDATA(FRACFOCUS = True, COGCC_SQL = True, COGCC_SHP = True):
    pathname = path.dirname(argv[0])
    adir = path.abspath(pathname)

    #Frac Focus
    if FRACFOCUS:
        url = 'https://www.fracfocusdata.org/digitaldownload/FracFocusCSV.zip'
        if path.exists(path.split(url)[-1]):
            remove(path.split(url)[-1])    

        download_and_extract_zip(url, dest_dir=getcwd())
        
        #filename = filename_from_request(url)
        #with ZipFile(filename, 'r') as zipObj:
           # Extract all the contents of zip file in current directory
        #   zipObj.extractall('FRAC_FOCUS')

    # COGCC SQLITE
    # https://dnrftp.state.co.us/
    if COGCC_SQL:
        url = 'https://dnrftp.state.co.us/COGCC/Temp/Gateway/CO_3_2.1.zip'
               
        if path.exists(path.split(url)[-1]):
            remove(path.split(url)[-1]) 
        download_and_extract_zip(url, 
                                 dest_dir=path.join(getcwd(),'COLORADO_SQL'))

        #filename = filename_from_request(url)
        #with ZipFile(filename, 'r') as zipObj:
           # Extract all the contents of zip file in current directory
        #   try:
        #       zipObj.extractall('COOGC_SQL')
        #   except BadZipfile:
        #       pass

        files = []        
        start_dir = path.join(adir,'COLORADO_SQL')
        pattern   = r'CO_3_2.*.sq.*'
        for dir,_,_ in walk(start_dir):
            files.extend(glob(path.join(getcwd(),'COLORADO_SQL')))

        shutil.move(files[0], path.join(adir, path.basename(files[0])))
        shutil.rmtree(path.join(adir,'COLORADO_SQL'))

    # COGCC shapefiles
    if COGCC_SHP:
        # url = 'https://cogcc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_LINES_SHP.ZIP' # NEW URL BELOW
        url = 'https://ecmc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_LINES_SHP.ZIP'       
        if path.exists(path.split(url)[-1]):
            remove(path.split(url)[-1]) 

        download_and_extract_zip(url, dest_dir=getcwd())
        filename = filename_from_request(url)
        #with ZipFile(filename, 'r') as zipObj:
        #   # Extract all the contents of zip file in current directory
        #   zipObj.extractall()
        remove(filename)

        # url = 'https://cogcc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_LINES_PENDING_SHP.ZIP' # NEW URL BELOW
        url = 'https://ecmc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_LINES_PENDING_SHP.ZIP'
        if path.exists(path.split(url)[-1]):
            remove(path.split(url)[-1])        
        download_and_extract_zip(url, dest_dir=getcwd())
        filename = wget.download(url)
        #with ZipFile(filename, 'r') as zipObj:
        #   # Extract all the contents of zip file in current directory
        #   zipObj.extractall()
        remove(filename)

        url = 'https://cogcc.state.co.us/documents/data/downloads/gis/WELLS_SHP.ZIP'
        url = 'https://ecmc.state.co.us/documents/data/downloads/gis/WELLS_SHP.ZIP'
        if path.exists(path.split(url)[-1]):
            remove(path.split(url)[-1]) 
        download_and_extract_zip(url, dest_dir=getcwd())
        filename = wget.download(url)
        #with ZipFile(filename, 'r') as zipObj:
        #   # Extract all the contents of zip file in current directory
        #   zipObj.extractall()
        remove(filename)

def _wait_for_download_complete(download_dir: Path, timeout: int = 120) -> Path:
    """
    Wait for a new file to appear and finish downloading.
    Works for Chrome/Edge (.crdownload / .tmp).
    Returns the completed file path.
    """
    download_dir = Path(download_dir)
    start = time.time()

    # snapshot existing files
    before = {p.name for p in download_dir.glob("*")}

    while True:
        # any new files?
        now_files = list(download_dir.glob("*"))
        new_files = [p for p in now_files if p.name not in before]

        # ignore partials
        partials = [p for p in now_files if p.suffix.lower() in {".crdownload", ".tmp", ".part"}]

        # if we have at least one new non-partial file and no partials are growing
        completed = [p for p in new_files if p.suffix.lower() not in {".crdownload", ".tmp", ".part"}]

        if completed and not partials:
            # choose the most recent completed file
            return max(completed, key=lambda p: p.stat().st_mtime)

        if time.time() - start > timeout:
            raise TimeoutError(f"Download did not complete within {timeout}s in {download_dir}")

        time.sleep(0.5)


def _safe_click(driver, element, timeout: int = 20):
    WebDriverWait(driver, timeout).until(EC.element_to_be_clickable(element))
    element.click()


def _extract_rows_from_docs_table(html: str) -> pd.DataFrame:
    """
    Parse the first table on the ECMC docs page and add a LINK column containing
    href for anchors whose visible text is 'Download'.
    """
    soup = BS(html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame()

    table = tables[0]
    dfs = pd.read_html(str(table), header=0)
    if not dfs:
        return pd.DataFrame()

    df = dfs[0].copy()

    # build list of hrefs for each Download anchor in DOM order
    download_anchors = table.find_all("a", string=lambda s: isinstance(s, str) and s.strip().lower() == "download")
    hrefs = [a.get("href") for a in download_anchors]

    # ECMC tables usually have a "Download" column; align by rows where Download == "Download"
    if "Download" in df.columns:
        mask = df["Download"].astype(str).str.strip().str.lower().eq("download")
        df["LINK"] = None
        # best-effort alignment: assign hrefs in order to masked rows
        idxs = list(df.index[mask])
        for i, ridx in enumerate(idxs):
            if i < len(hrefs):
                df.at[ridx, "LINK"] = hrefs[i]
    else:
        df["LINK"] = None

    return df

class CheckpointWriter:
    """
    Very lightweight checkpointing:
      - appends completed UWI10 lines to a text file (one per line)
      - thread-safe
    On restart you can load the file and skip those UWIs.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def load_completed(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            return {line.strip() for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()}
        except Exception:
            return set()

    def mark_done(self, uwi10: str) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(str(uwi10).strip() + "\n")

                
def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", str(s).strip())
    return s[:180] if len(s) > 180 else s

def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", errors="replace")

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def _sniff_kind_bytes(b: bytes) -> str:
    # --- Fast binary signatures ---
    if b.startswith(b"%PDF"):
        return "PDF"
    if b[:2] == b"PK":
        return "ZIP"
    if b[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIF"

    # --- Text detection (LAS is text) ---
    try:
        t = b.decode("utf-8", errors="ignore").upper()
    except Exception:
        return "UNKNOWN"

    # Strip leading whitespace, nulls, BOMs
    t = t.lstrip("\x00 \t\r\n")

    # LAS markers can appear later, not just first 4k
    if "~V" in t or "~VERSION" in t or "WRAP." in t or "VERS." in t:
        return "LAS"

    if t.startswith("<!DOCTYPE HTML") or t.startswith("<HTML"):
        return "HTML"

    return "UNKNOWN"


def _guess_ext(kind: str) -> str:
    return {"LAS": ".las", "PDF": ".pdf", "TIF": ".tif", "ZIP": ".zip"}.get(kind, "")

def _to_uwi10(x) -> str:
    s = re.sub(r"\D", "", str(x).strip())
    if len(s) == 9:
        s = "0" + s
    s = s[-10:]
    if len(s) == 10 and not s.startswith("05"):
        s = "05" + s[2:]
    return s

def _cowell_from_uwi10(uwi10: str) -> str:
    return str(uwi10)[2:10]  # 0512332615 -> 12332615

def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk_size), b""):
            h.update(b)
    return h.hexdigest()

def _files_identical(a: Path, b: Path) -> bool:
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
    except FileNotFoundError:
        return False
    return _sha256(a) == _sha256(b)

def _unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    k = 1
    while True:
        cand = p.with_name(f"{stem}_{k}{suf}")
        if not cand.exists():
            return cand
        k += 1


def _normalize_allowed_kinds(allowed_kinds) -> set[str]:
    if allowed_kinds is None:
        return {"LAS"}
    # If user passed a single string like "LAS", treat as one token
    if isinstance(allowed_kinds, str):
        allowed_kinds = [allowed_kinds]
    return {str(k).upper().strip() for k in allowed_kinds if str(k).strip()}


def _finalize_download_with_dupes(
    tmp_path: Path,
    final_path: Path,
    *,
    dupe_dirname: str = "DUPES",
) -> Path:
    """
    Moves tmp_path -> final_path, but if final_path exists:
      - if identical: delete tmp and return final_path
      - else: move tmp into DUPES/ with same filename (unique suffixed if needed)
    Returns the path where the downloaded file ended up.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists():
        if _files_identical(final_path, tmp_path):
            tmp_path.unlink(missing_ok=True)
            return final_path

        dupe_dir = final_path.parent / dupe_dirname
        dupe_dir.mkdir(parents=True, exist_ok=True)
        dupe_target = _unique_path(dupe_dir / final_path.name)
        tmp_path.replace(dupe_target)
        return dupe_target

    tmp_path.replace(final_path)
    return final_path

# ----------------------------
# Parse docs table
# ----------------------------

def _extract_docs_table(page_html: str, page_url: str) -> pd.DataFrame:
    soup = BS(page_html, "lxml")
    
    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame()

    target = None
    for t in tables:
        ths = [th.get_text(strip=True) for th in t.find_all("th")]
        if "Document Identifier" in ths and "Download" in ths:
            target = t
            break
    if target is None:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in target.find_all("th")]

    rows = []
    for tr in target.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        cells = [td.get_text(" ", strip=True) for td in tds]
        row = dict(zip(headers, cells))

        a = tr.find("a", string=re.compile(r"^\s*Download\s*$", re.I))
        href = a.get("href") if a else None
        row["DOWNLOAD_HREF"] = href
        row["DOWNLOAD_URL"] = urljoin(page_url, href) if href else None

        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# Requests session from Selenium
# ----------------------------

def _requests_session_from_selenium(driver) -> requests.Session:
    s = requests.Session()

    # Use browser UA and language headers
    try:
        ua = driver.execute_script("return navigator.userAgent;")
    except Exception:
        ua = None
    if ua:
        s.headers["User-Agent"] = ua

    # A few headers can matter for “looks like browser”
    s.headers.setdefault("Accept", "*/*")
    s.headers.setdefault("Accept-Language", "en-US,en;q=0.9")
    s.headers.setdefault("Connection", "keep-alive")

    # Transfer cookies
    for c in driver.get_cookies():
        s.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))

    return s


def _make_pooled_retry_session(base_session: requests.Session | None = None) -> requests.Session:
    """
    Adds connection pooling + retries for GETs (good for many doc pages + downloads).
    Safe to call on an existing session (keeps cookies/headers).
    """
    s = base_session or requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)

    # These help speed + reduce bytes
    # Avoid brotli ("br") because requests may not decode it, which breaks sniffing.
    s.headers["Accept-Encoding"] = "gzip, deflate"
    s.headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")

    return s


def _fetch_docs_html_fast(session: requests.Session, docurl: str, *, timeout_page: int = 20) -> str:
    """
    Fetch docs HTML via requests (FAST) instead of Selenium rendering.
    If table is present in server HTML, this is much faster than selenium.get().
    """
    # (connect_timeout, read_timeout)
    timeout = (10, timeout_page)
    r = session.get(docurl, timeout=timeout, allow_redirects=True)
    # If blocked or bad status, still return text for debug/inspection
    # but generally we want to treat non-200 as failure
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} fetching docs page")
    return r.text

def _kind_hint_from_url(url: str) -> str | None:
    u = (url or "").lower()
    if "downloaddocumentpdf" in u:
        return "PDF"
    if "downloaddocumenttif" in u or "downloaddocumenttiff" in u:
        return "TIF"
    if "downloaddocumentzip" in u:
        return "ZIP"
    if "download" in u and u.endswith(".las"):
        return "LAS"
    return None


def _kind_hint_from_content_type(ct: str | None) -> str | None:
    if not ct:
        return None
    ct = ct.lower()
    if "pdf" in ct:
        return "PDF"
    if "tif" in ct or "tiff" in ct:
        return "TIF"
    if "zip" in ct:
        return "ZIP"
    # LAS sometimes comes as text/plain or application/octet-stream; we can't trust this.
    return None
# ----------------------------
# Download (fast)
# ----------------------------

def _download_one(
    session: requests.Session,
    url: str,
    out_dir: Path,
    *,
    referer: Optional[str],
    timeout: int,
    allowed_kinds: set[str],
    name_hint: str,
    dupe_dirname: str = "DUPES",
    debug: bool = False,
) -> tuple[bool, dict]:
    info = {
        "url": url,
        "final_url": None,
        "status": None,
        "content_type": None,
        "sniffed_kind": None,
        "saved_as": None,
        "bytes": 0,
        "note": None,
    }

    
    headers = {}
    if referer:
        headers["Referer"] = referer

    try:
        r = session.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True)
        info["status"] = r.status_code
        info["final_url"] = str(r.url)
        info["content_type"] = r.headers.get("Content-Type")
    except Exception as ex:
        info["note"] = f"requests.get failed: {ex}"
        return False, info

    if r.status_code != 200:
        info["note"] = f"HTTP {r.status_code}"
        return False, info

    # Peek first chunk to sniff
    try:
        first = next(r.iter_content(chunk_size=8192))
    except StopIteration:
        info["note"] = "empty response"
        return False, info
    except Exception as ex:
        info["note"] = f"iter_content failed: {ex}"
        return False, info

    kind = _sniff_kind_bytes(first)
    info["sniffed_kind"] = kind

    if kind == "HTML":
        info["note"] = "got HTML (blocked page / needs cookies or referer)"
        return False, info

    if kind not in allowed_kinds:
        info["note"] = f"kind {kind} not in allowed_kinds={sorted(allowed_kinds)}"
        return False, info

    ext = _guess_ext(kind) or ""
    out_dir.mkdir(parents=True, exist_ok=True)

    final = out_dir / f"{name_hint}{ext}"
    tmp = out_dir / f"{name_hint}{ext}.part"

    # Write to tmp
    nbytes = 0
    try:
        with tmp.open("wb") as f:
            f.write(first)
            nbytes += len(first)
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
                    nbytes += len(chunk)
    except Exception as ex:
        info["note"] = f"write failed: {ex}"
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, info

    info["bytes"] = nbytes

    # --- DUPE HANDLING ---
    if final.exists():
        try:
            # If identical, drop tmp and skip
            if _files_identical(final, tmp):
                tmp.unlink(missing_ok=True)
                info["saved_as"] = str(final)
                info["note"] = "SKIPPED_IDENTICAL (already exists)"
                return True, info

            # Different content -> move into DUPES with same name
            dupe_dir = out_dir / dupe_dirname
            dupe_dir.mkdir(parents=True, exist_ok=True)
            dupe_target = dupe_dir / final.name
            dupe_target = _unique_path(dupe_target)

            tmp.replace(dupe_target)
            info["saved_as"] = str(dupe_target)
            info["note"] = f"SAVED_DUPE (existing differed; moved to {dupe_dirname})"
            return True, info

        except Exception as ex:
            info["note"] = f"dupe-check failed: {ex}"
            # fall back: don't overwrite; keep tmp as unique dupe
            dupe_dir = out_dir / dupe_dirname
            dupe_dir.mkdir(parents=True, exist_ok=True)
            dupe_target = _unique_path(dupe_dir / final.name)
            try:
                tmp.replace(dupe_target)
                info["saved_as"] = str(dupe_target)
                info["note"] = f"SAVED_DUPE_FALLBACK ({ex})"
                return True, info
            except Exception as ex2:
                info["note"] = f"failed to save dupe fallback: {ex2}"
                return False, info

    # Normal: no existing file
    try:
        tmp.replace(final)
    except Exception as ex:
        info["note"] = f"finalize failed: {ex}"
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, info

    info["saved_as"] = str(final)
    return True, info


class TokenBucket:
    """
    Global bandwidth limiter shared across threads.
    rate_bytes_per_sec: refill rate
    capacity_bytes: max burst
    """
    def __init__(self, rate_bytes_per_sec: int, capacity_bytes: int | None = None):
        self.rate = float(rate_bytes_per_sec)
        self.capacity = float(capacity_bytes if capacity_bytes is not None else rate_bytes_per_sec)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, nbytes: int):
        """Block until nbytes tokens available."""
        n = float(nbytes)
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.updated
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                    self.updated = now

                if self.tokens >= n:
                    self.tokens -= n
                    return

                # not enough tokens; compute wait
                missing = n - self.tokens
                wait = missing / self.rate if self.rate > 0 else 0.1

            time.sleep(min(max(wait, 0.005), 0.25))

            
def _ext_from_content_disposition(cd: str | None) -> str | None:
    """
    Extract extension from Content-Disposition filename, if present.
    """
    if not cd:
        return None
    # e.g. Content-Disposition: attachment; filename="ABC.las"
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, flags=re.I)
    if not m:
        return None
    fname = m.group(1).strip()
    suf = Path(fname).suffix.lower()
    if suf in (".las", ".pdf", ".tif", ".tiff", ".zip"):
        return ".tif" if suf == ".tiff" else suf
    return None

def _is_probably_text(b: bytes) -> bool:
    # crude but effective: LAS is almost entirely ASCII-ish text
    if not b:
        return False
    # allow tabs/newlines/carriage returns + printable ascii
    good = sum((c in b"\t\r\n") or (32 <= c <= 126) for c in b)
    return (good / max(len(b), 1)) > 0.92


def _sniff_kind_bytes_las_friendly(b: bytes) -> str:
    # binary signatures
    if b.startswith(b"%PDF"):
        return "PDF"
    if b[:2] == b"PK":
        return "ZIP"
    if b[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIF"

    # if it looks like HTML
    bb = b.lstrip()
    if bb.lower().startswith(b"<!doctype html") or bb.lower().startswith(b"<html"):
        return "HTML"

    # LAS: text + LAS section markers somewhere early
    if _is_probably_text(b):
        t = b.decode("utf-8", errors="ignore").upper()
        # common LAS section markers
        if "~V" in t or "~VERSION" in t or "~W" in t or "~WELL" in t or "~A" in t:
            return "LAS"

    return "UNKNOWN"


def _download_one_managed(
    session: requests.Session,
    url: str,
    base_path: Path,  # no extension
    *,
    referer: str | None,
    allowed_kinds: set[str],
    timeout_http: int = 60,
    max_retries: int = 3,
    backoff_base: float = 0.8,
    limiter: TokenBucket | None = None,
    chunk_size: int = 256 * 1024,
) -> dict:
    info = {
        "url": url,
        "final_url": None,
        "status": None,
        "kind": None,
        "saved_as": None,
        "bytes": 0,
        "note": None,
        "ok": False,
        "referer": referer,
    }

    headers = {}
    if referer:
        headers["Referer"] = referer

    base_path.parent.mkdir(parents=True, exist_ok=True)
    req_timeout = (15, timeout_http)

    for attempt in range(max_retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=req_timeout, allow_redirects=True) as r:
                info["status"] = r.status_code
                info["final_url"] = str(r.url)

                if r.status_code != 200:
                    info["note"] = f"HTTP {r.status_code}"
                    raise RuntimeError(info["note"])

                it = r.iter_content(chunk_size=chunk_size)

                # Read a larger "sniff window" so LAS markers (~V etc) are likely included
                sniff_buf = b""
                while len(sniff_buf) < 64 * 1024:  # 64KB sniff
                    chunk = next(it, b"")
                    if not chunk:
                        break
                    sniff_buf += chunk

                if not sniff_buf:
                    info["note"] = "empty response"
                    raise RuntimeError(info["note"])

                # Use Content-Disposition extension if present; otherwise sniff-based
                cd_ext = _ext_from_content_disposition(r.headers.get("Content-Disposition"))
                kind = _sniff_kind_bytes_las_friendly(sniff_buf)
                info["kind"] = kind

                if kind == "HTML":
                    info["note"] = "got HTML (blocked/auth page)"
                    raise RuntimeError(info["note"])

                if kind not in allowed_kinds:
                    info["note"] = f"SKIP (sniffed {kind}, allowed={sorted(allowed_kinds)})"
                    return info

                sniff_ext = _guess_ext(kind) or ""
                ext = cd_ext or sniff_ext or ".bin"

                final_path = base_path.with_suffix(ext)
                tmp_path = final_path.with_suffix(final_path.suffix + ".part")

                # Write sniffed bytes + remainder of stream
                with tmp_path.open("wb") as f:
                    if limiter is not None:
                        limiter.consume(len(sniff_buf))
                    f.write(sniff_buf)
                    info["bytes"] += len(sniff_buf)

                    for chunk in it:
                        if not chunk:
                            continue
                        if limiter is not None:
                            limiter.consume(len(chunk))
                        f.write(chunk)
                        info["bytes"] += len(chunk)

                saved = _finalize_download_with_dupes(tmp_path, final_path, dupe_dirname="DUPES")
                info["saved_as"] = str(saved)
                info["ok"] = True
                info["note"] = "DOWNLOADED"
                return info

        except Exception as ex:
            if info["note"] is None:
                info["note"] = str(ex)

        if attempt < max_retries:
            time.sleep(backoff_base * (2 ** attempt) + random.random() * 0.3)

    return info

def download_many_managed(
    base_session: requests.Session,
    items: list[tuple[str, Path, str | None, str]],  # (url, out_path, referer, uwi10)
    *,
    allowed_kinds: set[str],
    max_workers: int = 6,
    bandwidth_limit_mb_s: float | None = None,
    timeout_http: int = 60,
    progress_every: int = 200,  # print every N completed downloads
) -> list[dict]:
    """
    Thread-safe: each worker thread gets its own Session cloned from base_session (cookies+headers),
    but they share the same TokenBucket limiter for global bandwidth control.
    """

    limiter = None
    if bandwidth_limit_mb_s is not None:
        rate = int(bandwidth_limit_mb_s * 1024 * 1024)
        limiter = TokenBucket(rate_bytes_per_sec=rate, capacity_bytes=rate)

    # Thread-local sessions
    tls = threading.local()

    def _clone_session() -> requests.Session:
        s = requests.Session()

        # copy headers
        s.headers.update(base_session.headers)

        # copy cookies
        s.cookies.update(base_session.cookies)

        # mount adapters (pool + retries) like your pooled session
        retry = Retry(
            total=5, connect=5, read=5,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)

        # helpful defaults
        # Avoid brotli for same reason
        s.headers["Accept-Encoding"] = "gzip, deflate"
        s.headers.setdefault("Connection", "keep-alive")

        return s

    def _get_thread_session() -> requests.Session:
        s = getattr(tls, "session", None)
        if s is None:
            tls.session = _clone_session()
            s = tls.session
        return s

    def _wrap(url: str, out_path: Path, ref: str | None, uwi10: str) -> dict:
        s = _get_thread_session()
        res = _download_one_managed(
            s, url, out_path,
            referer=ref,
            allowed_kinds=allowed_kinds,
            timeout_http=timeout_http,
            limiter=limiter,
        )
        res["uwi"] = uwi10
        return res

    results: list[dict] = []
    total = len(items)
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_wrap, url, out_path, ref, uwi10) for (url, out_path, ref, uwi10) in items]

        per_future_timeout = timeout_http + 45

        for f in as_completed(futs):
            try:
                res = f.result(timeout=per_future_timeout)
            except FutTimeout:
                res = {"ok": False, "note": "future timeout (hung worker)"}
            except Exception as ex2:
                res = {"ok": False, "note": f"future raised: {ex2}"}

            results.append(res)
            done += 1

            # ultra-light progress (prints rarely)
            if progress_every and (done % progress_every == 0 or done == total):
                dt = max(time.time() - t0, 1e-6)
                rate = done / dt
                print(f"[download] {done}/{total} complete  (~{rate:.1f}/s)  ok={sum(1 for r in results if r.get('ok'))}")

    return results


def Get_LAS(
    UWIS: Iterable[str | int | float],
    *,
    url_base: str = "https://ecmc.state.co.us/cogisdb/Resources/Docs?ClassID=07&ID=",
    logs_folder: str | Path = "LOGS",
    only_class: str = "Well Logs",
    allowed_kinds: Optional[set[str]] = None,   # {"LAS"} or {"LAS","PDF","TIF","ZIP"}
    # ---------- performance controls ----------
    collect_workers: int = 16,                  # doc-page fetch+parse concurrency
    download_workers: int = 6,                  # file download concurrency
    bandwidth_limit_mb_s: float | None = 10.0,  # global cap; set None to disable
    timeout_page: int = 20,                     # doc-page read timeout (requests)
    timeout_http: int = 90,                     # download read timeout
    headless_ok: bool = True,
    debug: bool = False,
    debug_dir: str | Path = "LAS_DEBUG",
    max_files_per_uwi: Optional[int] = None,
    # ---------- progress + restart ----------
    progress_every_wells: int = 10,             # print every N completed wells
    checkpoint_file: str | Path | None = "LAS_DEBUG/completed_uwis.txt",
    resume: bool = True,
):
    """
    Stable solution:
      Selenium once → parallel requests doc fetch → parse → one global download pool.

    Returns:
        downloaded_files: list[Path]
        bad_links: list[str]
        report: list[dict]
    """
    # normalize inputs
    if isinstance(UWIS, (str, float, int)):
        UWIS = [UWIS]
    UWIS_10 = [_to_uwi10(x) for x in UWIS]

    allowed_kinds = _normalize_allowed_kinds(allowed_kinds)

    base_dir = Path.cwd()
    logs_dir = (base_dir / Path(logs_folder)).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    debug_root = (base_dir / Path(debug_dir)).resolve()
    if debug or checkpoint_file:
        debug_root.mkdir(parents=True, exist_ok=True)

    # checkpointing
    cp = None
    completed_set: set[str] = set()
    if checkpoint_file is not None:
        cp_path = Path(checkpoint_file)
        if not cp_path.is_absolute():
            cp_path = (base_dir / cp_path).resolve()
        cp = CheckpointWriter(cp_path)
        if resume:
            completed_set = cp.load_completed()

    # filter UWIs if resuming
    if resume and completed_set:
        UWIS_10 = [u for u in UWIS_10 if u not in completed_set]

    downloaded_files: list[Path] = []
    bad_links: list[str] = []
    report: list[dict] = []

    # Selenium ONCE: establish cookies/UA for a requests session
    browser = get_driver(download_dir=str(logs_dir), headless=(headless_ok and not debug))
    try:
        sess = _requests_session_from_selenium(browser)
        sess = _make_pooled_retry_session(sess)  # add pooling + retries

        # --------------------------
        # Phase A: collect all jobs
        # --------------------------
        # We'll build:
        #   all_jobs: list[(download_url, base_path_no_ext, referer_docurl)]
        #   per_well_report: store counts + notes
        all_jobs: list[tuple[str, Path, str | None, str]] = []  # (url, base_path, referer, uwi10)
        report_by_uwi: dict[str, dict] = {}
        expected_jobs_by_uwi: dict[str, int] = {}

        # minimal progress without overhead
        done_lock = threading.Lock()
        done_count = 0
        total_wells = len(UWIS_10)

        def collect_one(uwi10: str) -> tuple[str, dict, list[tuple[str, Path, str | None]]]:
            cowell = _cowell_from_uwi10(uwi10)
            docurl = f"{url_base}{cowell}"

            uwi_rep = {
                "uwi": uwi10,
                "docurl": docurl,
                "rows_total": 0,
                "rows_after_class_filter": 0,
                "download_jobs": 0,
                "notes": [],
                "artifacts": {},
            }

            # fetch docs HTML via requests
            try:
                html = _fetch_docs_html_fast(sess, docurl, timeout_page=timeout_page)
            except Exception as ex:
                uwi_rep["notes"].append(f"docs fetch failed: {ex}")
                return uwi10, uwi_rep, []

            if debug:
                # html dump only (cheap); avoid screenshots
                htm = debug_root / f"{uwi10}_docs.html"
                _write_text(htm, html)
                uwi_rep["artifacts"]["html"] = str(htm)

            # parse HTML table
            try:
                df = _extract_docs_table(html, docurl)
            except Exception as ex:
                uwi_rep["notes"].append(f"extract table failed: {ex}")
                return uwi10, uwi_rep, []

            uwi_rep["rows_total"] = int(df.shape[0])
            if df.empty:
                uwi_rep["notes"].append("no rows extracted")
                return uwi10, uwi_rep, []

            # filter class
            if "Class" in df.columns and only_class:
                want = only_class.strip().lower()
                df = df[df["Class"].astype(str).str.strip().str.lower().eq(want)].copy()

            uwi_rep["rows_after_class_filter"] = int(df.shape[0])
            if df.empty:
                uwi_rep["notes"].append(f"no rows after Class == {only_class!r}")
                return uwi10, uwi_rep, []

            # require URLs
            df = df[df["DOWNLOAD_URL"].astype(str).str.startswith("http", na=False)].copy()
            if max_files_per_uwi is not None:
                df = df.head(int(max_files_per_uwi)).copy()

            jobs: list[tuple[str, Path, str | None]] = []
            for _, row in df.iterrows():
                dl_url = row.get("DOWNLOAD_URL")
                if not isinstance(dl_url, str) or not dl_url.strip():
                    continue

                doc_num = row.get("Document Number")
                doc_name = row.get("Document Name")
                doc_date = row.get("Date")

                hint_parts = [uwi10]
                if isinstance(doc_date, str) and doc_date.strip():
                    hint_parts.append(_safe_name(doc_date))
                if isinstance(doc_num, str) and doc_num.strip():
                    hint_parts.append(_safe_name(doc_num))
                if isinstance(doc_name, str) and doc_name.strip():
                    hint_parts.append(_safe_name(doc_name)[:80])

                hint = "_".join(hint_parts)
                base_path = logs_dir / hint  # no extension yet
                jobs.append((dl_url.strip(), base_path, docurl, uwi10))

            uwi_rep["download_jobs"] = len(jobs)

            return uwi10, uwi_rep, jobs

        with ThreadPoolExecutor(max_workers=collect_workers) as ex:
            futs = [ex.submit(collect_one, u) for u in UWIS_10]
            for f in as_completed(futs):
                uwi10, uwi_rep, jobs = f.result()

                report_by_uwi[uwi10] = uwi_rep
                expected_jobs_by_uwi[uwi10] = len(jobs)
                all_jobs.extend(jobs)

                # If a well has zero jobs, it's truly "done" immediately
                if cp is not None and len(jobs) == 0:
                    cp.mark_done(uwi10)

                # lightweight progress
                with done_lock:
                    done_count += 1
                    if progress_every_wells and (done_count % progress_every_wells == 0 or done_count == total_wells):
                        # prints one line occasionally; negligible overhead
                        print(f"[collect] {done_count}/{total_wells} wells complete (last: {uwi10})  jobs_total={len(all_jobs)}")

        # build report in original order
        for u in UWIS_10:
            report.append(report_by_uwi.get(u, {"uwi": u, "notes": ["missing report"]}))

        # de-dupe exact (url, base_path) pairs BUT KEEP uwi for regroup/checkpoint
        seen = set()
        deduped: list[tuple[str, Path, str | None, str]] = []
        for url, p, ref, uwi in all_jobs:
            key = (url, str(p))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((url, p, ref, uwi))

        all_jobs = deduped

        if debug:
            _write_json(debug_root / "collect_summary.json", {
                "wells": len(UWIS_10),
                "jobs": len(all_jobs),
                "allowed_kinds": sorted(allowed_kinds),
            })

        if not all_jobs:
            return [], bad_links, report

        if debug and all_jobs:
            test_url, test_path, test_ref, test_uwi = all_jobs[0]
            print(f"[debug] smoke test first download uwi={test_uwi} url={test_url}")
            try:
                r = sess.get(test_url, headers={"Referer": test_ref} if test_ref else None,
                             stream=True, timeout=(15, 30), allow_redirects=True)
                b = next(r.iter_content(chunk_size=4096), b"")
                print(f"[debug] smoke status={r.status_code} ct={r.headers.get('Content-Type')} sniff={_sniff_kind_bytes(b)} first16={b[:16]!r}")
            except Exception as ex:
                print(f"[debug] smoke test failed: {ex}")

        # --------------------------
        # Phase B: global download pool
        # --------------------------
        # NOTE: download_many_managed uses your TokenBucket limiter globally.
        # It currently returns list[dict]. We’ll also add minimal progress prints
        # without adding per-chunk overhead.

        # Wrap your existing downloader to add occasional progress prints without slowing downloads.
        results: list[dict] = []
        total_jobs = len(all_jobs)
        dl_done = 0
        dl_lock = threading.Lock()

        # sanity check
        print(f"[download] starting: {len(all_jobs)} jobs, allowed={sorted(allowed_kinds)}, workers={download_workers}")
        print(f"[download] example url: {all_jobs[0][0]}")
        print(f"[download] starting {len(all_jobs)} downloads with workers={download_workers}, timeout_http={timeout_http}s")

        # Use your existing download_many_managed if you prefer.
        # BUT: it currently builds its own ThreadPool and limiter; that’s fine.
        # We keep it and just print every so often after futures complete.
        results = download_many_managed(
            sess,
            all_jobs,
            allowed_kinds=allowed_kinds,
            max_workers=download_workers,
            bandwidth_limit_mb_s=bandwidth_limit_mb_s,
            timeout_http=timeout_http,
            progress_every=25,   # print occasionally even with 2 workers
        )
        # Collect results -> downloaded_files/bad_links
        for res in results:
            if res.get("ok") and res.get("saved_as"):
                downloaded_files.append(Path(res["saved_as"]))
            else:
                if res.get("url"):
                    bad_links.append(str(res["url"]))

        # -------- checkpoint after downloads complete per well --------
        if cp is not None:
            ok_jobs_by_uwi: dict[str, int] = {}

            for res in results:
                u = res.get("uwi")
                if not u:
                    continue
                if res.get("ok"):
                    ok_jobs_by_uwi[u] = ok_jobs_by_uwi.get(u, 0) + 1

            # Mark done only if all jobs succeeded for that UWI
            for u, n_expected in expected_jobs_by_uwi.items():
                if n_expected <= 0:
                    # already handled during collection, but safe
                    continue
                if ok_jobs_by_uwi.get(u, 0) >= n_expected:
                    cp.mark_done(u)
                    

        if debug:
            ok_n = sum(1 for r in results if r.get("ok"))
            print(f"[download] ok={ok_n}/{len(results)}  workers={download_workers}  cap={bandwidth_limit_mb_s} MB/s")
            _write_json(debug_root / "download_results.json", results)
            _write_json(debug_root / "final_report.json", report)

        return downloaded_files, bad_links, report

    finally:
        with contextlib.suppress(Exception):
            browser.quit()
 
def Get_ProdData(UWIs,file='prod_data.db',SQLFLAG=0, PROD_DATA_TABLE = 'PRODDATA', PROD_SUMMARY_TABLE = 'PRODUCTION_SUMMARY', FOLDER = 'PRODDATA', RETURN_DATA = False):
    #if 1==1:
    #URL_BASE = 'https://cogcc.state.co.us/cogis/ProductionWellMonthly.asp?APICounty=XCOUNTYX&APISeq=XNUMBERX&APIWB=XCOMPLETIONX&Year=All'
    URL_BASE = 'https://cogcc.state.co.us/production/?&apiCounty=XCOUNTYX&apiSequence=XNUMBERX'
    URL_BASE = 'https://ecmc.state.co.us/cogisdb/Facility/Production?api_county_code=XCOUNTYX&api_seq_num=XNUMBERX'
    #pathname = path.dirname(argv[0])
    #adir = path.abspath(pathname)
    adir = getcwd()
    #warnings.simplefilter("ignore")
    OUTPUT=pd.DataFrame(columns=['BTU_MEAN','BTU_STD'
                                 ,'API_MEAN','API_STD'
                                 ,'Peak_Oil_Date','Peak_Oil_Days','Peak_Oil_CumOil','Peak_Oil_CumGas','Peak_Oil_CumWtr'
                                 ,'Peak_Gas_Date','Peak_Gas_Days','Peak_Gas_CumOil','Peak_Gas_CumGas','Peak_Gas_CumWtr'
                                 ,'OWR_PrePeakOil','OWR_PostPeakGas'
                                 ,'GOR_PrePeakOil','GOR_PeakGas','GOR_PostPeakGOR'
                                 ,'WOC_PostPeakOil','WOC_PostPeakGas'
                                 ,'GOR_Final','OWC_Final'
                                 ,'Month1'
                                 ,'GOR_MO2to4','GOR_MO5to7','GOR_MO11to13','GOR_MO23to25','GOR_MO35to37','GOR_MO47to49'
                                 ,'OWR_MO2to4','OWR_MO5to7','OWR_MO11to13','OWR_MO23to25','OWR_MO35to37','OWR_MO47to49'
                                 ,'Production_Formation'])
    MonthArray = np.arange(3,49,3)
    for i in MonthArray:
        OUTPUT[str(i)+'Mo_CumOil'] = np.nan
        OUTPUT[str(i)+'Mo_CumGas'] = np.nan
        OUTPUT[str(i)+'Mo_CumWtr'] = np.nan

    if len(UWIs[0])<=1:
        UWIs=[UWIs]
        print(UWIs[0])

    UWIs = [x for x in UWIs if x[0:2]=='05']

    PRODDATA = pd.DataFrame()
    ct = 0
    t1 = perf_counter()
    for UWI in UWIs:
        UWI = WELLAPI(UWI).STRING(10)
        #print(UWI)              
        if (floor(ct/20)*20) == ct:
            print(str(ct)+' of '+str(len(UWIs)))
        ct+=1
        html = soup = pdf = None 

        ERROR=0

        while ERROR == 0: #if 1==1:
            connection_attempts = 4 
            #Screen for Colorado wells
            userows=pd.DataFrame()
            if UWI[:2] == '05':
                #print(UWI)
                #Reduce well to county and well numbers
                COWELL=UWI[5:10]
                COWELL64 = base64.b64encode(str(COWELL).encode('ascii')).decode().replace('=','')
                COUNTYCODE64 = base64.b64encode(str(UWI[2:5]).encode('ascii')).decode().replace('=','')      
                if len(UWI)>=12:
                    COMPLETION=UWI[10:12]
                else:
                    COMPLETION="00"
                COMPLETIONCODE64 = base64.b64encode(str(COMPLETION).encode('ascii')).decode().replace('=','')                           
                #docurl=re.sub('XNUMBERX',COWELL64,URL_BASE)
                #docurl=re.sub('XCOUNTYX',COUNTYCODE64,docurl)
                #docurl=re.sub('XCOMPLETIONX',str(UWI[2:5]),docurl)
                
                docurl=re.sub('XNUMBERX',COWELL,URL_BASE)
                docurl=re.sub('XCOUNTYX',str(UWI[2:5]),docurl)
                docurl=re.sub('XCOMPLETIONX',COMPLETION,docurl)
                
                #try:
                #    html = urlopen(docurl).read()
                #except Exception as ex:
                #    print(f'Error connecting to {docurl}.')
                #    ERROR=1
                #    continue
                #soup = BS(html, 'lxml')
                #try:
                #    parsed_table = soup.find_all('table')[1]
                #except:
                #    print(f'No Table for {UWI}.')
                #    ERROR=1
                #    continue
                if perf_counter() - t1 < 0.5:
                    sleep(0.5)
                t1 = perf_counter()
                
                try:
                    #pdf = pd.read_html(docurl,encoding='utf-8', header=0)[1]
                    content = requests_retry_session().get(docurl).content
                    s_cont = str(content)
                    if bool(re.search('no records found',s_cont,re.I)):
                        print('No production data at ' + docurl)
                        ERROR = 1
                        continue  
                    rawData = pd.read_html(StringIO(content.decode('utf-8')))
                    pdf = rawData[-1]
                except:
                    print(f'Error connecting to {docurl}.')
                    ERROR=1
                    continue
                #pdf=pd.read_html('https://cogcc.state.co.us/cogis/ProductionWellMonthly.asp?APICounty=123&APISeq=42282&APIWB=00&Year=All')[1]
                try:
                    SEQ      = pdf.iloc[:,pdf.keys().str.contains('.*SEQUENCE.*', regex=True, case=False,na=False)].keys()[0]
                except:
                    for i in range(0,pdf.shape[1]):
                        newcol = str(pdf.keys()[i])+'_'+'_'.join(pdf.iloc[0:8,i].astype('str'))
                        pdf=pdf.rename({pdf.keys()[i]:newcol},axis=1)
                    try:    
                        SEQ      = pdf.iloc[:,pdf.keys().str.contains('.*SEQUENCE.*', regex=True, case=False,na=False)].keys()
                        x = min(np.array(pdf.loc[pdf[SEQ].astype(str) == COWELL,SEQ].index))-1   # non-value indexes
                        xrows = list(range(0, x))
                        for i in range(0,pdf.shape[1]):
                            newcol = str(pdf.keys()[i])+'_'+'_'.join(pdf.iloc[0:x,i].astype('str'))
                            pdf = pdf.rename({pdf.keys()[i]:newcol},axis=1)
                            pdf = pdf.drop(xrows,axis=0)
                    except:
                        print(f'Cannot parse tabels 1 at: {docurl}.')
                        ERROR = 1
                        continue

##                DATE     =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*FIRST.*MONTH.*', regex=True, case=False,na=False)].keys()[0])
##                DAYSON   =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*DAYS.*PROD.*', regex=True, case=False,na=False)].keys()[0])
##                OIL      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*OIL.*PROD.*', regex=True, case=False,na=False)].keys()[0])
##                GAS      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*GAS.*PROD.*', regex=True, case=False,na=False)].keys()[0])
##                WTR      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*WATER.*VOLUME.*', regex=True, case=False,na=False)].keys()[0])
##                API      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*OIL.*GRAVITY.*', regex=True, case=False,na=False)].keys()[0])
##                BTU      =pdf.keys().get_loc(pdf.iloc[0,pdf.keys().str.contains('.*GAS.*BTU.*', regex=True, case=False,na=False)].keys()[0])
                try: 
                    DATE     = pdf.iloc[:,pdf.keys().str.contains('.*FIRST.*MONTH.*', regex=True, case=False,na=False)].keys()[0]
                    DAYSON   = pdf.iloc[0,pdf.keys().str.contains('.*DAYS.*PROD.*', regex=True, case=False,na=False)].keys()[0]
                    OIL      = pdf.iloc[0,pdf.keys().str.contains('.*OIL.*PROD.*', regex=True, case=False,na=False)].keys()[0]
                    GAS      = pdf.iloc[0,pdf.keys().str.contains('.*GAS.*PROD.*', regex=True, case=False,na=False)].keys()[0]
                    WTR      = pdf.iloc[0,pdf.keys().str.contains('.*WATER.*VOLUME.*', regex=True, case=False,na=False)].keys()[0]
                    API      = pdf.iloc[0,pdf.keys().str.contains('.*OIL.*GRAVITY.*', regex=True, case=False,na=False)].keys()[0]
                    BTU      = pdf.iloc[0,pdf.keys().str.contains('.*GAS.*BTU.*', regex=True, case=False,na=False)].keys()[0]
                    FM       = pdf.iloc[0,pdf.keys().str.contains('.*Formation.*', regex=True, case=False,na=False)].keys()[0]
                except:
                    print(f'Cannot parse tabels 2 at: {docurl}.')
                    ERROR = 1
                    continue                    
                    
                # Date is date formatted                
                pdf[DATE]=pd.to_datetime(pdf[DATE]).dt.date
                # Sort on earliest date first
                pdf.sort_values(by = [DATE],inplace = True)
                pdf.index = range(1, len(pdf) + 1)
               
                PRODOIL = pdf[OIL].dropna().index
                PRODGAS = pdf[GAS].dropna().index
                PRODWTR = pdf[WTR].dropna().index
           
                pdf['OIL_RATE'] = pdf[OIL]/pdf[DAYSON]
                pdf['GAS_RATE'] = pdf[GAS]/pdf[DAYSON]
                pdf['WTR_RATE'] = pdf[WTR]/pdf[DAYSON]
                pdf['PROD_DAYS'] = pdf[DAYSON].cumsum()
                      
                pdf['CUMOIL'] = pdf[OIL].cumsum()
                pdf['CUMGAS'] = pdf[GAS].cumsum()
                pdf['CUMWTR'] = pdf[WTR].cumsum()
                
                pdf[['TMB_OIL','TMB_GAS','TMB_WTR']] = np.nan
           
                pdf.loc[PRODOIL,'TMB_OIL'] = pdf.loc[PRODOIL,'CUMOIL'] / pdf.loc[PRODOIL,OIL]
                pdf.loc[PRODGAS,'TMB_GAS'] = pdf.loc[PRODGAS,'CUMGAS'] / pdf.loc[PRODGAS,GAS]
                pdf.loc[PRODWTR,'TMB_WTR'] = pdf.loc[PRODWTR,'CUMWTR'] / pdf.loc[PRODWTR,WTR]      

                pdf[['GOR','OWR','WOR','OWC','WOC']] = np.nan
                               
                pdf.loc[PRODOIL,'GOR'] = pdf[GAS]*1000/pdf[OIL]
                pdf.loc[PRODWTR,'OWR'] = pdf[OIL]/pdf[WTR]
                pdf.loc[PRODOIL,'WOR'] = pdf[WTR]/pdf[OIL]
                
                m = PRODOIL.join(PRODWTR,how='outer')
                pdf.loc[m,'OWC'] = pdf.loc[m,OIL]/(pdf.loc[m,WTR]+pdf.loc[m,OIL])
                pdf.loc[m,'WOC'] = pdf.loc[m,WTR]/(pdf[WTR]+pdf.loc[m,OIL])

                if pdf[[API]].dropna(how='any').shape[0]>3:
                    OUTPUT.at[UWI,'API_MEAN']         = pdf[API].astype('float').describe()[1]
                    OUTPUT.at[UWI,'API_STD']          = pdf[API].astype('float').describe()[2]

                if pdf[[BTU]].dropna(how='any').shape[0]>3:
                    OUTPUT.at[UWI,'BTU_MEAN']         = pdf[BTU].astype('float').describe()[1]
                    OUTPUT.at[UWI,'BTU_STD']          = pdf[BTU].astype('float').describe()[2]

                if pdf[[OIL,GAS]].dropna(how='any').shape[0]>3:
                    OUTPUT.at[UWI,'Peak_Oil_Date']   = pdf[DATE][pdf[OIL].idxmax()]
                    OUTPUT.at[UWI,'Peak_Oil_Days']   = pdf['PROD_DAYS'][pdf[OIL].idxmax()]
                    OUTPUT.at[UWI,'Peak_Oil_CumOil'] = pdf[OIL][0:pdf[OIL].idxmax()].sum()
                    OUTPUT.at[UWI,'Peak_Oil_CumGas'] = pdf[GAS][0:pdf[OIL].idxmax()].sum()

                    OUTPUT.at[UWI,'Peak_Gas_Date']   = pdf[DATE][pdf[GAS].idxmax()]
                    OUTPUT.at[UWI,'Peak_Gas_Days']   = pdf['PROD_DAYS'][pdf[GAS].idxmax()]
                    OUTPUT.at[UWI,'Peak_Gas_CumOil'] = pdf[OIL][0:pdf[GAS].idxmax()].sum()
                    OUTPUT.at[UWI,'Peak_Gas_CumGas'] = pdf[GAS][0:pdf[GAS].idxmax()].sum()

                    PREPEAKOIL  = pdf.loc[(pdf['PROD_DAYS']-pdf['PROD_DAYS'][pdf[OIL].idxmax()]).between(-100,0),:].index
                    POSTPEAKOIL = pdf.loc[(pdf['PROD_DAYS'][pdf[OIL].idxmax()]-pdf['PROD_DAYS']).between(0,100),:].index
                    POSTPEAKGAS = pdf.loc[(pdf['PROD_DAYS'][pdf[GAS].idxmax()]-pdf['PROD_DAYS']).between(0,100),:].index
                    PEAKGAS = pdf.loc[(pdf['PROD_DAYS'][pdf[GAS].idxmax()]-pdf['PROD_DAYS']).between(-50,50),:].index
                    LATEWATER = pdf.index[pdf['TMB_WTR']>200]
                    LATEGAS = pdf.index[pdf['TMB_GAS']>30]
                    
                    if pdf.loc[PREPEAKOIL,OIL].sum()>0:
                        OUTPUT.at[UWI,'GOR_PrePeakOil']  = pdf.loc[PREPEAKOIL,GAS].sum() * 1000 / pdf.loc[PREPEAKOIL,OIL].sum()
                    if pdf.loc[PEAKGAS,OIL].sum()>0:
                        OUTPUT.at[UWI,'GOR_PeakGas']     = pdf.loc[PEAKGAS,GAS].sum() * 1000 / pdf.loc[PEAKGAS,OIL].sum()
                                       
                    if len(PRODOIL.intersection(PRODGAS).intersection(PRODWTR)) >3 : 
                        if pdf.loc[PREPEAKOIL,WTR].sum()>0:
                            OUTPUT.at[UWI,'OWR_PrePeakOil']  = pdf.loc[PREPEAKOIL,OIL].sum()/pdf.loc[PREPEAKOIL,WTR].sum()
                        if pdf.loc[POSTPEAKGAS,WTR].sum() >0:
                            OUTPUT.at[UWI,'OWR_PostPeakGas'] = pdf.loc[POSTPEAKGAS,OIL].sum()/pdf.loc[POSTPEAKGAS,WTR].sum()     
                        
                        OUTPUT.at[UWI,'WOC_PostPeakOil'] = pdf.loc[POSTPEAKOIL,WTR].sum() / (pdf.loc[POSTPEAKOIL,WTR].sum()+pdf.loc[POSTPEAKOIL,OIL].sum())
                        OUTPUT.at[UWI,'WOC_PostPeakGas'] = pdf.loc[POSTPEAKGAS,WTR].sum() / (pdf.loc[POSTPEAKGAS,WTR].sum()+pdf.loc[POSTPEAKGAS,OIL].sum())        
                        OUTPUT.at[UWI,'Peak_Oil_CumWtr'] = pdf[WTR][0:pdf[OIL].idxmax()].sum()
                        OUTPUT.at[UWI,'Peak_Gas_CumWtr'] = pdf[WTR][0:pdf[GAS].idxmax()].sum()
                      
                        if len(LATEGAS)>3:
                            OUTPUT.at[UWI,'GOR_Final'] = pdf.loc[LATEGAS, GAS].sum() / pdf.loc[LATEGAS, OIL].sum() * 1000
                        if len(LATEWATER)>3:
                            OUTPUT.at[UWI,'OWC_Final'] =  pdf.loc[LATEWATER, OIL].sum() / (pdf.loc[LATEWATER, OIL].sum()+pdf.loc[LATEWATER, WTR].sum())

                    # Emily uses Month 1 begins at 1st month w/ +14days oil prod
                    if len(pdf[DATE].dropna())>10:
                        MONTH1 = pdf.loc[(pdf[DAYSON]>14) & (pdf[OIL]>0),DATE].min()
                        OUTPUT.at[UWI,'Month1'] = MONTH1

                        if not isinstance(MONTH1,float):
                            pdf['EM_PRODMONTH'] = (pd.to_datetime(pdf[DATE]).dt.year - MONTH1.year)*12+(pd.to_datetime(pdf[DATE]).dt.month - MONTH1.month)+1

                            for i in MonthArray:
                                if max(pdf['EM_PRODMONTH']) >= i:
                                    i_dwn = i-1
                                    i_up = i+1
                                    OUTPUT.at[UWI,str(i)+'Mo_CumOil'] = pdf.loc[(pdf['EM_PRODMONTH']<=i),OIL].sum()
                                    OUTPUT.at[UWI,str(i)+'Mo_CumGas'] = pdf.loc[(pdf['EM_PRODMONTH']<=i),GAS].sum()
                                    OUTPUT.at[UWI,str(i)+'Mo_CumWtr'] = pdf.loc[(pdf['EM_PRODMONTH']<=i),WTR].sum()
                                            
                                    if pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum() > 0:
                                        OUTPUT.at[UWI,'GOR_MO'+str(i_dwn)+'to'+str(i_up)]  = pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum()
                                    if (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),WTR].sum()) >=1:   
                                        OUTPUT.at[UWI,'OWC_MO'+str(i)] = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),WTR].sum())
                                    if pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),WTR].sum() > 0: 
                                        OUTPUT.at[UWI,'OWR_MO'+str(i_dwn)+'to'+str(i_up)]  = pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),WTR].sum()
                    OUTPUT.at[UWI,'CUM_OIL'] = pdf[OIL].sum()
                    OUTPUT.at[UWI,'CUM_GAS'] = pdf[GAS].sum()
                    OUTPUT.at[UWI,'CUM_WATER'] = pdf[WTR].sum()
                    
            OUTPUT.at[UWI,'Production_Formation'] = '_'.join(pdf[FM].unique())

            pdf['UWI'] = UWI
            PRODDATA = pd.concat([PRODDATA,pdf],axis=0,join='outer',ignore_index=True) 

            ERROR = 1
            
    OUTPUT=OUTPUT.dropna(how='all')
    OUTPUT.index.name = 'UWI'   
    OUTPUT.reset_index(inplace = True)
    if not OUTPUT.empty:
        OUTPUT = DF_UNSTRING(OUTPUT)
    OUTPUT['UWI10'] = OUTPUT.UWI.apply(lambda x: WELLAPI(x).API2INT(10))
           
    SQL_COLS = '''([UWI] INTEGER PRIMARY KEY
         ,[BTU_MEAN] REAL
         ,[BTU_STD] REAL
         ,[API_MEAN] REAL
         ,[API_STD] REAL
         ,[Peak_Oil_Date] DATE
         ,[Peak_Oil_Days] INTEGER
         ,[Peak_Oil_CumOil] REAL
         ,[Peak_Oil_CumGas] REAL
         ,[Peak_Gas_Date] DATE
         ,[Peak_Gas_Days] INTEGER
         ,[Peak_Gas_CumOil] REAL
         ,[Peak_Gas_CumGas] REAL
         ,[OWR_PrePeakOil] REAL
         ,[OWR_PostPeakGas] REAL
         ,[WOC_PrePeakOil] REAL
         ,[WOC_PostPeakOil] REAL
         ,[WOC_PostPeakGas] REAL
         ,[Peak_Oil_CumWtr] REAL
         ,[Peak_Gas_CumWtr] REAL
         ,[Month1] DATE
         ,[GOR_MO2to4] REAL
         ,[GOR_MO5to7] REAL
         ,[GOR_MO11to13] REAL
         ,[GOR_MO23to25] REAL
         ,[GOR_MO35to37] REAL
         ,[GOR_MO47to49] REAL
         ,[OWR_MO2to4] REAL
         ,[OWR_MO5to7] REAL
         ,[OWR_MO11to13] REAL
         ,[OWR_MO23to25] REAL
         ,[OWR_MO35to37] REAL
         ,[OWR_MO47to49] REAL
         ,[OWC_MO3] REAL
         ,[OWC_MO6] REAL
         ,[OWC_MO12] REAL
         ,[OWC_MO24] REAL
         ,[OWC_MO36] REAL
         ,[OWC_MO48] REAL
         ,[Production_Formation] TEXT
         ,[3Mo_CumOil] REAL
         ,[6Mo_CumOil] REAL
         ,[9Mo_CumOil] REAL
         ,[12Mo_CumOil] REAL
         ,[15Mo_CumOil] REAL
         ,[18Mo_CumOil] REAL
         ,[21Mo_CumOil] REAL
         ,[24Mo_CumOil] REAL
         ,[27Mo_CumOil] REAL
         ,[30Mo_CumOil] REAL
         ,[33Mo_CumOil] REAL
         ,[36Mo_CumOil] REAL
         ,[39Mo_CumOil] REAL
         ,[42Mo_CumOil] REAL
         ,[45Mo_CumOil] REAL
         ,[48Mo_CumOil] REAL
         ,[3Mo_CumGas] REAL
         ,[6Mo_CumGas] REAL
         ,[9Mo_CumGas] REAL
         ,[12Mo_CumGas] REAL
         ,[15Mo_CumGas] REAL
         ,[18Mo_CumGas] REAL
         ,[21Mo_CumGas] REAL
         ,[24Mo_CumGas] REAL
         ,[27Mo_CumGas] REAL
         ,[30Mo_CumGas] REAL
         ,[33Mo_CumGas] REAL
         ,[36Mo_CumGas] REAL
         ,[39Mo_CumGas] REAL
         ,[42Mo_CumGas] REAL
         ,[45Mo_CumGas] REAL
         ,[48Mo_CumGas] REAL
         ,[3Mo_CumWtr] REAL
         ,[6Mo_CumWtr] REAL
         ,[9Mo_CumWtr] REAL
         ,[12Mo_CumWtr] REAL
         ,[15Mo_CumWtr] REAL
         ,[18Mo_CumWtr] REAL
         ,[21Mo_CumWtr] REAL
         ,[24Mo_CumWtr] REAL
         ,[27Mo_CumWtr] REAL
         ,[30Mo_CumWtr] REAL
         ,[33Mo_CumWtr] REAL
         ,[36Mo_CumWtr] REAL
         ,[39Mo_CumWtr] REAL
         ,[42Mo_CumWtr] REAL
         ,[45Mo_CumWtr] REAL
         ,[48Mo_CumWtr] REAL
         ,[CUM_OIL] REAL
         ,[CUM_GAS] REAL
         ,[CUM_WATER] REAL)
         '''

    #PROD_DATA_TABLE = 'PRODDATA', 
    #PROD_SUMMARY_TABLE = 'PRODUCTION_SUMMARY'    
    
    print('Saving Results')
    PRODDATA = DF_UNSTRING(PRODDATA)
    PRODDATA[DATE] = pd.to_datetime(PRODDATA[DATE])
    PROD_FNAME = 'PRODUCTION_'+str(PRODDATA['UWI'].iloc[0])+'_'+str(PRODDATA['UWI'].iloc[-1])+'_'+datetime.datetime.now().strftime('%Y%m%d')
    PRODDATA.columns = PRODDATA.columns.str.replace(' ','_')
    PRODDATA['UWI10'] = PRODDATA.UWI.apply(lambda x: WELLAPI(x).API2INT(10))

    if not path.exists(path.join(adir,FOLDER)):
        mkdir(path.join(adir,FOLDER))
               
    PRODDATA.to_parquet(path.join(adir,FOLDER,PROD_FNAME+'.parquet') )

    PRODDATA.reset_index(drop = True, inplace = True)
           
    if (OUTPUT.shape[0] > 0) & (SQLFLAG != 0):
        conn = sqlite3.connect(file)
        c = conn.cursor()    
           
        COLTYPES = FRAME_TO_SQL_TYPES(OUTPUT)

        if not PROD_SUMMARY_TABLE in LIST_SQL_TABLES(conn):
            INIT_SQL_TABLE(conn, PROD_SUMMARY_TABLE, COLTYPES)
              
        OLD = pd.read_sql('SELECT * FROM {0} LIMIT 100'.format(PROD_SUMMARY_TABLE), conn)

        if OLD.shape[0]>0:
           OLD_COLTYPES = FRAME_TO_SQL_TYPES(OLD)
           COLTYPES.update(OLD_COLTYPES)
           
        INIT_SQL_TABLE(conn, PROD_SUMMARY_TABLE, COLTYPES)

        MISSING_COLS = [k for k in COLTYPES.keys() if k.upper() not in OUTPUT.keys().str.upper()]
        if len(MISSING_COLS)>0:
            OUTPUT[MISSING_COLS] = None   

        SUCCESS = 0
        COUNT = -1
        while SUCCESS == 0 and COUNT < 1000:
            COUNT+=1     

            try:
                #c.execute('CREATE TABLE IF NOT EXISTS ' + PROD_SUMMARY_TABLE + ' ' + SQL_COLS)
                tmp = str(OUTPUT.index.max())
                OUTPUT.to_sql(tmp, conn, if_exists='replace', index = False)
                SQL_CMD='DELETE FROM {0} WHERE [UWI] IN (SELECT [UWI] FROM \'{1}\');'.format(PROD_SUMMARY_TABLE,tmp)
                c.execute(SQL_CMD)
                
                SQL_CMD = 'DELETE FROM {0} WHERE UWI IN ({1})'.format(PROD_SUMMARY_TABLE,str(OUTPUT.UWI.tolist())[1:-1])
                c.execute(SQL_CMD)
                conn.commit()                
                
                SQL_CMD = 'INSERT INTO {0} SELECT * FROM \'{1}\';'.format(PROD_SUMMARY_TABLE,tmp)
                c.execute(SQL_CMD)

                SQL_CMD = 'DROP TABLE \'{0}\';'.format(tmp)
                c.execute(SQL_CMD)
                conn.commit()
                SUCCESS = 1
                print('PROD SUMMARY SAVED')
                break
            except Exception as e: 
                print(e)
                sleep(10)
                pass
           
            if SUCCESS == 0:
                try:
                    #c.execute('CREATE TABLE IF NOT EXISTS ' + PROD_SUMMARY_TABLE + ' ' + SQL_COLS)
                    tmp = str(OUTPUT.index.max())
                    OUTPUT.to_sql(tmp, conn, if_exists='replace', index = True)
                    SQL_CMD='DELETE FROM '+PROD_SUMMARY_TABLE+' WHERE [UWI] IN (SELECT [UWI] FROM \''+tmp+'\');'
                    c.execute(SQL_CMD)
                    SQL_CMD ='INSERT INTO '+PROD_SUMMARY_TABLE+' SELECT * FROM \''+tmp+'\';'
                    c.execute(SQL_CMD)
                    conn.commit()

                    SQL_CMD = 'DROP TABLE \''+tmp+'\';'
                    c.execute(SQL_CMD)
                    conn.commit()
                    print('PROD SUMMARY SAVED')
                    SUCCESS = 1
                    break
                except Exception as e: 
                    print(e)
                    pass
           
        #LOAD PRODUCTION DATA
        SUCCESS = 0
        COUNT = -1

        COLTYPES = FRAME_TO_SQL_TYPES(PRODDATA)
           
        if not PROD_DATA_TABLE in LIST_SQL_TABLES(conn):
            INIT_SQL_TABLE(conn, PROD_DATA_TABLE, COLTYPES)
           
        OLD = pd.read_sql('SELECT * FROM {0} LIMIT 100'.format(PROD_DATA_TABLE), conn)
        if OLD.shape[0]>0:
           OLD_COLTYPES = FRAME_TO_SQL_TYPES(OLD)
           COLTYPES.update(OLD_COLTYPES)
        INIT_SQL_TABLE(conn, PROD_DATA_TABLE, COLTYPES)

        MISSING_COLS = [k for k in COLTYPES.keys() if k.upper() not in PRODDATA.keys().str.upper()]
        if len(MISSING_COLS)>0:
            PRODDATA[MISSING_COLS] = None   

        while SUCCESS == 0 and COUNT < 1000:
            COUNT += 1
            try:
                tmp = str(PRODDATA.index.max()) 
                PRODDATA.to_sql(tmp, conn, if_exists='replace', index = False)
                SQL_CMD = 'DELETE FROM \'{0}\' WHERE rowid IN (SELECT A.rowid FROM \'{0}\' A INNER JOIN \'{1}\' B ON A.API_Sequence=B.API_Sequence AND A.First_of_Month = B.First_of_Month AND A.Formation = B.Formation);'.format(PROD_DATA_TABLE, tmp)
                c.execute(SQL_CMD)
                SQL_CMD ='INSERT INTO {0} SELECT * FROM \'{1}\';'.format(PROD_DATA_TABLE,tmp)
                c.execute(SQL_CMD)
                SQL_CMD = 'DROP TABLE \'{0}\';'.format(tmp)
                conn.commit()  
                print('PROD DATA SAVED')
                SUCCESS =1
                break
            except Exception as e: 
                print(e)
                sleep(10)
                      
    try:
        conn.close()
    except:
        pass
               
    if RETURN_DATA == False:
        return (OUTPUT)
    elif RETURN_DATA == True:
        return (OUTPUT, PRODDATA)
    else:
        return (OUTPUT)


def Get_Scouts(UWIs, db=None, TABLE_NAME = 'CO_SCOUT'):
    # PASSING ERROR if True:
           
    Strings = ['WELL NAME/NO', 'OPERATOR', 'STATUS DATE','FACILITYID','COUNTY','LOCATIONID','LAT/LON','ELEVATION',
               'SPUD DATE','JOB DATE','JOB END DATE','TOP PZ','BOTTOM HOLE LOCATION',#r'COMPLETED.*INFORMATION.*FORMATION',
               'TOTAL FLUID USED','MAX PRESSURE','TOTAL GAS USED','FLUID DENSITY','TYPE OF GAS',
               'NUMBER OF STAGED INTERVALS','TOTAL ACID USED','MIN FRAC GRADIENT','RECYCLED WATER USED',
               'TOTAL FLOWBACK VOLUME','PRODUCED WATER USED','TOTAL PROPPANT USED',
               'TUBING SIZE','TUBING SETTING DEPTH','# OF HOLES','INTERVAL TOP','INTERVAL BOTTOM',r'^HOLE SIZE','FORMATION NAME','1ST PRODUCTION DATE',
               'BBLS_H2O','BBLS_OIL','CALC_GOR', 'GRAVITY_OIL','BTU_GAS','TREATMENT SUMMARY']
    
    Strings = ['WELL NAME/NO', 'OPERATOR', 'STATUS DATE','FACILITYID','COUNTY','LOCATIONID','LAT/LON','ELEVATION',
               'SPUD DATE','JOB DATE','JOB END DATE','TOP PZ','BOTTOM HOLE LOCATION',#r'COMPLETED.*INFORMATION.*FORMATION',
               'TOTAL FLUID USED','MAX PRESSURE','TOTAL GAS USED','FLUID DENSITY',
               'NUMBER OF STAGED INTERVALS','TOTAL ACID USED','MIN FRAC GRADIENT','RECYCLED WATER USED',
               'TOTAL FLOWBACK VOLUME','PRODUCED WATER USED','TOTAL PROPPANT USED',
               'TUBING SIZE','TUBING SETTING DEPTH','# OF HOLES','INTERVAL TOP','INTERVAL BOTTOM',r'^HOLE SIZE','FORMATION NAME','1ST PRODUCTION DATE',
               'TREATMENT SUMMARY']
           
    status_pat = re.compile(r'Status:([\sA-Z]*)[0-9]{1,2}/[0-9]{1,2}/[0-9]{1,2}', re.I)

    OUTPUT=[]
    pagedf=[]
    xSummary = None
    URL_BASE = 'https://cogcc.state.co.us/cogis/FacilityDetail.asp?facid=XNUMBERX&type=WELL'
    URL_BASE = 'https://ecmc.state.co.us/cogisdb/Facility/FacilityDetailExpand?api=XNUMBERX'
           
    pathname = path.dirname(argv[0])
    adir = path.abspath(pathname)

    dir_add = path.join(adir,'SCOUTS')
    if path.isdir(dir_add) == False:
        mkdir(dir_add)

    warnings.simplefilter("ignore")
           
    if isinstance(UWIs,(int,str,float)):
        UWIs=[UWIs]
    elif isinstance(UWIs,list) == False:
        UWIs = list(UWIs)

    UWIs = [WELLAPI(x).STRING(10) for x in UWIs]
    UWIs = list(set(UWIs))
           
    for UWI in UWIs:
        #if 1==1:
        UWI = WELLAPI(UWI).STRING(10)
        #if len(UWI)%2 == 1:
        #    UWI = UWI.zfill(len(UWI)+1)

        print(UWI+" "+datetime.datetime.now().strftime("%d/%m/%Y_%H:%M:%S"))
        docurl = None
        connection_attempts = 4 
        #Screen for Colorado wells
        userows=pd.DataFrame()
        if UWI[:2] == '05':
            #Reduce well to county and well numbers
            docurl=re.sub('XNUMBERX',UWI[2:10],URL_BASE)
            RETRY=0
            
            while RETRY<8:
                try:
                    pagedf=pd.read_html(docurl)
                    status_code = requests.get(docurl).status_code
                    RETRY=60
                except:
                    pagedf=[]
                    RETRY += 1
                    sleep(10)

        xSummary = pd.DataFrame()
        if len(pagedf)>0:
            page_dict = {}
            for i,j in enumerate(pagedf):
                page_dict.update({i:j})
                
            page_merge = pd.DataFrame()
            
            for s in page_dict:
                page_merge = pd.concat([page_merge,page_dict[s]])
                
            page_merge.reset_index(drop=True, inplace = True)
            LOCS = Find_Str_Locs(page_merge,Strings)
            mm = (LOCS[['Columns','Rows']].applymap(len)!=0).product(axis =1).replace(0,np.nan).dropna().index
            LOCS = LOCS.loc[mm,:]
            
            for i in LOCS.index:
                T = str(LOCS.loc[i,'Title']).upper()
                C = LOCS.loc[i,'Columns']
                R = LOCS.loc[i,'Rows']
                score0 = 0
                score1 = -1
                R0 = 0
                C0 = 0
                for j,c in enumerate(C):
                    TEST_STR = str(page_merge.iloc[R[j],C[j]]).upper()
                    if TEST_STR == str(np.nan).upper():
                        continue
                    if str(page_merge.iloc[R[j],1+C[j]]) == str(np.nan):
                           continue
                    score1 = difflib.SequenceMatcher(None, T, TEST_STR).ratio()
                    if score1 > score0:
                        score0 = score1
                        R0 = R[j]
                        C0 = c
                    elif (score1>0) * (score1 == score0) * (len(TEST_STR)< len(str(page_merge.iloc[R0,C0]))):
                        score0 = score1
                        R0 = R[j]
                        C0 = c
                        
                xSummary.at[UWI,T] = page_merge.iloc[R0,C0+1]
                   
            #xSummary = Summarize_Page(pagedf,Strings) 
            xSummary['UWI']=UWI

            # Status code
            #STAT_CODE = None
            #try:
            #    status = status_pat.search(page_merge.iloc[1,0])
            #    status = status.group(1)
            #    STAT_CODE = status.strip()
            #except:
            #    print('status error')
            #    STAT_CODE = 'ERR'
            #    pass

            xSummary.at[UWI,'WELL_STATUS'] = str(status_code) 
            

            #xSummary = pd.DataFrame([xSummary.values],columns= xSummary.index.tolist())

            if type(OUTPUT)==list:
                OUTPUT=xSummary
            else:
                OUTPUT = pd.concat([OUTPUT,xSummary], axis = 0, ignore_index=True)
                #OUTPUT=OUTPUT.append(xSummary,ignore_index=True)
    try:     
        KEYS = list(OUTPUT.keys()) 
    except:
        print(f'no OUTPUT for {UWIs[0]}:{UWIs[-1]}')
        return None
    KEYS = [re.sub(r'[^0-9a-zA-Z]','_',k) for k in KEYS]
    KEYS = [re.sub(r'1ST','FIRST',k) for k in KEYS]
    OUTPUT.columns = KEYS
    OUTPUT['UWI10'] = OUTPUT.UWI.apply(lambda x:WELLAPI(x).API2INT(10))       

    FILENAME = str(UWIs[0])+'_'+str(UWIs[-1])+"_"+datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    FILENAME = path.join(dir_add,FILENAME) 
    try:
        DF_UNSTRING(OUTPUT).to_json(FILENAME+'.JSON')
        DF_UNSTRING(OUTPUT).to_parquet(FILENAME+'.PARQUET')
    except:
        print("CANNOT UNSTRING "+FILENAME)

    if db != None:
        DATECOLS = [col for col in OUTPUT.columns if 'DATE' in col.upper()]
        for k in DATECOLS:
            OUTPUT.loc[:,k]=pd.to_datetime(OUTPUT.loc[:,k]).fillna(np.nan)
            #OUTPUT.loc[OUTPUT.loc[:,k],k]       
     
        SCHEMA = FRAME_TO_SQL_TYPES(OUTPUT)
        ATTEMPTS = -1
        while (ATTEMPTS < 100):      
            ATTEMPTS+=1
            try:
                conn = sqlite3.connect(db)
              
                INIT_SQL_TABLE(conn, TABLE_NAME, FIELD_DICT= SCHEMA)

                # NEEDS CONVERSION OF PYTHON TYPES TO SQL TYPES
                #for k,v in OUTPUT.dtypes.to_dict().items():    
                #    SQL_COLS=SQL_COLS+'['+str(k)+'] '+str(v)+','
                #c.execute('CREATE TABLE IF NOT EXISTS ' + TABLE_NAME + ' ')
                #sql = "select * from %s where 1=0;" % table_name
                #c.execute(sql)
                #TBL_COLS = [d[0] for d in curs.description]
                #ADD_COLS = list(set(SQL_COLS).difference(TBL_COLS))
                SCOUTED_UWIS = pd.read_sql('SELECT DISTINCT UWI FROM {}'.format(TABLE_NAME),conn)
                
                if OUTPUT.UWI.isin(SCOUTED_UWIS).any():
                    SCOUT_DF =  pd.read_sql('SELECT * FROM {}'.format(TABLE_NAME),conn)
                    if 'index' in SCOUT_DF.keys():
                        SCOUT_DF.drop('index', axis =1, inplace = True)
                      
                    KEYS = list(SCOUT_DF.keys())
                    KEYS = [re.sub(r'[^0-9a-zA-Z]','_',k) for k in KEYS]
                    KEYS = [re.sub(r'1ST','FIRST',k) for k in KEYS]
                    SCOUT_DF.columns = KEYS
                    if (len(keys)-len(set(keys)))>0:
                        for k in keys:
                            test = keys.count(k)
                            if test > 1:           
                                m1 = SCOUT_DF[k].iloc[:,0].isna()
                                m2 = SCOUT_DF[k].iloc[:,1].isna()
                                idx1 = keys.index(k)
                                keys.reverse()
                                idx2 = len(keys) - keys.index(k)
                                keys.reverse()
                                SCOUT_DF.iloc[idx1,m1] = SCOUT_DF.iloc[idx2,m1]
                                idx_cols = list(np.arange(len(keys)))
                                del idx_cols[idx2]
                                SCOUT_DF = SCOUT_DF.iloc[:,idx_cols]
                    m = SCOUT_DF.UWI.isin(OUTPUT.UWI)
                    SCOUT_DF = pd.concat([SCOUT_DF.loc[~m],OUTPUT],axis=0, ignore_index=True)
                    SCOUT_DF['UWI10'] = SCOUT_DF.UWI.apply(lambda x:WELLAPI(x).API2INT(10)) 
                    SCOUT_DF.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
                else:
                    OUTPUT.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
                ATTEMPTS = 200
            except:
                sleep(30)
    return(OUTPUT)

def Merge_Frac_Focus(DIR = None, SAVE=False):
    # pathname = path.dirname(argv[0])
    adir = getcwd()

    if DIR == None:
        DIR = adir
    else:
        DIR = path.join(adir,DIR)
           
    #if 1==1:
    FLIST = filelist(SUBDIR = DIR, EXT='.csv',BEGIN = 'fracfocus')
    FracFocus = pd.DataFrame()
    for f in FLIST:
        if DIR!= None:
           f = path.join(DIR,f)
        freg_df = pd.read_csv(f,low_memory=False)
        #freg_df = freg_df.drop_duplicates()
        FracFocus = pd.concat([FracFocus,freg_df],axis=0,join='outer',ignore_index=True)
        FracFocus = FracFocus.drop_duplicates()
                      
    APILEN = int(((FracFocus.APINumber.astype(str).replace(r'~\d','',regex=True).apply(len)/2).apply(np.ceil)*2).max())             
    FracFocus.APINumber = FracFocus.APINumber.apply(lambda x:WELLAPI(x).API2INT(APILEN))                                                    
    FracFocus = DF_UNSTRING(FracFocus)
    if SAVE:
        FracFocus.to_json('FracFocusTables.JSON')
        FracFocus.to_parquet('FracFocusTables.PARQUET')
          
    return(FracFocus)
   
def SUMMARIZE_COGCC(SAVE = False, DB = 'FIELD_DATA.db',TABLE_NAME = 'COGCC_SQL_SUMMARY'):
    # SQLite COMMMANDS
    # well data
    Q1 = """
        SELECT
                P.StateProducingZoneKey
                ,P.ProductionDate
                ,P.StateProducingUnitKey
                ,P.DaysProducing
                ,P.Oil
                ,P.Gas
                ,P.Water
                ,SUM(CASE WHEN P.DaysProducing + 0 = 0 THEN 28
                           ELSE P.DaysProducing 
                           END) 
                    OVER(PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as DaysOn
                ,CAST(SUM(CASE WHEN P.DaysProducing + 0 = 0 THEN 28
                           ELSE P.DaysProducing
                           END) 
                  OVER(PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC)/30.4 AS int) AS MonthsOn
                ,SUM(P.Oil+0) OVER 
                        (PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as CumOil
                ,SUM(P.Gas+0) OVER 
                        (PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as CumGas        
                ,SUM(P.Water+0) OVER 
                        (PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as CumWater
                ,P.GAS / P.OIL *1000 as GOR
                ,P.OIL / IIF(P.DaysProducing is null,28,P.DaysProducing) AS OilRate_BOPD
                ,P.Gas / IIF(P.DaysProducing is null = 0,28,P.DaysProducing) AS GasRate_MCFPD
                ,P.Water / IIF(P.DaysProducing is null = 0,28,P.DaysProducing) AS WaterRate_BWPD
                ,((P.GAS / P.OIL - LAG(P.GAS,2) OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                        / LAG(P.OIL,2)  OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                                ) +
                        (P.GAS / P.OIL - LAG(P.GAS,1) OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                        / LAG(P.OIL,1)  OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                                ))*1000
                        AS dGOR
                ,PM.StartDate
                ,PM.PeakOil
                ,PM.PeakGas
                ,PM.PeakWater
                ,P.Oil/P.DaysProducing / PM.PeakOil AS NormOilRate
                ,P.Gas/P.DaysProducing / PM.PeakGas AS NormGasRate
                ,P.Water/P.DaysProducing / PM.PeakWater AS NormWater
                ,P.Gas /P.DaysProducing /  PM.PeakGas / (P.Oil / P.DaysProducing / PM.PeakOil) AS NormGOR
                ,(0+STRFTIME('%Y',PM.StartDate)) as MINYEAR
                ,W.*
        FROM Production P
        LEFT JOIN Well AS W ON W.StateWellKey = P.StateProducingUnitKey
        LEFT JOIN (
                select StateProducingUnitKey
                        , MIN(ProductionDate) as StartDate
                        , MAX(Oil/DaysProducing) AS PeakOil
                        , MAX(Gas/DaysProducing) AS PeakGas
                        ,MAX(Water/DaysProducing) AS PeakWater 
                        from PRODUCTION 
                        WHERE DaysProducing < 600
                        GROUP BY StateProducingUnitKey
                ) PM
                ON PM.StateProducingUnitKey = P.StateProducingUnitKey
        WHERE (P.OIL + P.WATER + P.GAS) >0 
        --      AND (STRFTIME('%Y',PM.StartDate)+0) > 2015
        --      AND substr(P.StateProducingUnitKey,1,7)+0 >=  512334
        --  AND (0+STRFTIME('%Y',MIN(ProductionDate) OVER (PARTITION BY StateProducingUnitKey)))>=2015
        --      AND Trim(P.StateProducingZoneKey) = "NBRR"
        -- GROUP BY StateProducingUnitKey
        ORDER BY P.StateProducingUnitKey,P.ProductionDate DESC
    """

    Q1 = """
    SELECT
        StateProducingUnitKey
        ,MIN(ProductionDate) as StartDate
        ,MAX(Oil/DaysProducing) AS PeakOil
        ,MAX(Gas/DaysProducing) AS PeakGas
        ,MAX(Water/DaysProducing) AS PeakWater
        FROM PRODUCTION 
        WHERE DaysProducing < 800
        GROUP BY StateProducingUnitKey
    """

    # Completions Groups
    Q2_Drop = "DROP TABLE If EXISTS temp_table1;"
    Q2 = """
    CREATE TABLE IF NOT EXISTS temp_table1 AS
        SELECT DISTINCT
            StateProducingUnitKey
            ,CAST(StateProducingUnitKey AS INTEGER) AS INT_SPUKEY
            ,ProductionDate
            ,SUM(Oil + Gas + Water) OVER (PARTITION BY  StateProducingUnitKey
                                                ORDER BY ProductionDate
                                                ROWS BETWEEN UNBOUNDED PRECEDING    
                                                    AND 1 PRECEDING) 
                AS Cum_OilGasWtr
        FROM Production
        WHERE (Oil + Gas + Water)>=0
        GROUP BY StateProducingUnitKey,ProductionDate;
    SELECT 
        S.StateWellKey
        ,S.StateStimulationKey
        ,S.TreatmentDate
        ,TR2.PastTreatmentDate
        ,TR2.PastStateStimulationKey
        ,P1.Cum_OilGasWtr as CumTreatOne
        ,P2.Cum_OilGasWtr as CumTreatTwo
        ,P1.Cum_OilGasWtr - P2.Cum_OilGasWtr as Cum_Difference
        , DATE(TR.TreatmentDate,'start of month') AS DATE1
        , DATE(TR2.PastTreatmentDate,'start of month') AS DATE2
        ,TR.TreatmentRank as TreatmentRank
    FROM Stimulation S
    LEFT JOIN (
        SELECT 
            StateStimulationKey
            ,StateWellKey
            ,TreatmentDate
            ,Rank() OVER (
                PARTITION BY StateWellKey
                ORDER BY TreatmentDate ASC) TreatmentRank
            ,COUNT(TreatmentDate) Over (PARTITION BY StateWellKey) as TreatmentCount
        FROM Stimulation 
        ) TR
            ON TR.StateStimulationKey = S.StateStimulationKey
    LEFT JOIN (
        SELECT 
            StateStimulationKey AS PastStateStimulationKey
            ,StateWellKey
            ,TreatmentDate As PastTreatmentDate
            ,Rank() OVER (
                PARTITION BY StateWellKey
                ORDER BY TreatmentDate ASC) PastTreatmentRank
        FROM Stimulation ) TR2
    ON  (TR2.PastTreatmentRank + 1)= TR.TreatmentRank AND TR2.StateWellKey=TR.StateWellKey          
    LEFT JOIN temp_table1 P1
        ON  P1.INT_SPUKEY = CAST(S.StateWellKey AS INTEGER)
            AND DATE(P1.ProductionDate,'start of month') = DATE(TR.TreatmentDate,'start of month')
    LEFT JOIN temp_table1 P2
        ON  P2.INT_SPUKEY = CAST(S.StateWellKey AS INTEGER)
            AND DATE(P2.ProductionDate,'start of month') = DATE(TR2.PastTreatmentDate,'start of month')
    ORDER BY Cum_Difference DESC
    """

    #Completion Volumes
    Q3 = """
    SELECT 
            S.*
            ,SF.Type
            ,SF.TotalAmount
            ,SF.QC_CountUnits
        FROM Stimulation S
        LEFT JOIN(
            SELECT 
                StateStimulationKey
                ,FluidType as Type
                ,SUM(Volume) as TotalAmount
                ,COUNT(DISTINCT FluidUnits) as QC_CountUnits
            FROM StimulationFluid
            GROUP BY StateStimulationKey,FluidType
            ) SF
            ON SF.StateStimulationKey = S.StateStimulationKey
    UNION   
    SELECT 
            S.*
            ,SP.Type
            ,SP.TotalAmount
            ,SP.QC_CountUnits
        FROM Stimulation S      
            LEFT JOIN(
            SELECT
                StateStimulationKey
                ,'PROPPANT' as Type
                ,SUM(ProppantAmount) as TotalAmount
                ,COUNT(DISTINCT ProppantUnits) as QC_CountUnits
            FROM StimulationProppant
            GROUP BY StateStimulationKey
            ) SP
            ON SP.StateStimulationKey = S.StateStimulationKey   
    """

    # 3,6,9,12,15,18,21,24 month cum
    Q4="""
    SELECT 
        PP.StateProducingUnitKey
        ,Min(CASE
        WHEN PP.Oil + PP.Gas + PP.Water >=0
        THEN PP.ProductionDate  
        Else NULL
        END)  as FirstProduction 
        ,Min(CASE 
            WHEN PP.OilRank=1
            THEN PP.CumOil
            Else NULL
            END
            ) AS PeakOil_CumOil
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.CumGas
            Else NULL
            END
            )  AS PeakOil_CumGas
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.CumWater
            Else NULL
            END
            )  AS PeakOil_CumWater 
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.ProductionDate
            Else NULL
            END
            )  AS PeakOil_Date
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.DaysOn
            Else NULL
            END
            )  AS PeakOil_DaysOn
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.OilRate_BBLPD
            Else NULL
            END
            )  AS PeakOil_OilRate
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.GasRate_MCFPD
            Else NULL
            END
            )  AS PeakOil_GasRate
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.WaterRate_BBLPD
            Else NULL
            END
            )  AS PeakOil_WaterRate
        ,Min(CASE 
            WHEN PP.WaterRank=1
            THEN PP.CumOil
            Else NULL
            END
            ) AS PeakWater_CumOil
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.CumGas
            Else NULL
            END
            )  AS PeakWater_CumGas
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.CumWater
            Else NULL
            END
            )  AS PeakWater_CumWater 
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.ProductionDate
            Else NULL
            END
            )  AS PeakWater_Date
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.DaysOn
            Else NULL
            END
            )  AS PeakWater_DaysOn
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.OilRate_BBLPD
            Else NULL
            END
            )  AS PeakWater_OilRate
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.GasRate_MCFPD
            Else NULL
            END
            )  AS PeakWater_GasRate
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.WaterRate_BBLPD
            Else NULL
            END
            )  AS PeakWater_WaterRate
        ,Min(CASE 
            WHEN PP.GasRank=1
            THEN PP.CumOil
            Else NULL
            END
            ) AS PeakGas_CumOil
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.CumGas
            Else NULL
            END
            )  AS PeakGas_CumGas
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.CumWater
            Else NULL
            END
            )  AS PeakGas_CumWater
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.ProductionDate
            Else NULL
            END
            )  AS PeakGas_Date
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.DaysOn
            Else NULL
            END
            )  AS PeakGas_DaysOn
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.OilRate_BBLPD
            Else NULL
            END
            )  AS PeakGas_OilRate
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.GasRate_MCFPD
            Else NULL
            END
            )  AS PeakGas_GasRate
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.WaterRate_BBLPD
            Else NULL
            END
            )  AS PeakGas_WaterRate    
         ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*3)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '3MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*6)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '6MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*9)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '9MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*12)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '12MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*15)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '15MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*18)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '18MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*21)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '21MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*24)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '24MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*3)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '3MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*6)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '6MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*9)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '9MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*12)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '12MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*15)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '15MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*18)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '18MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*21)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '21MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*24)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '24MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*3)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '3MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*6)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '6MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*9)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '9MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*12)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '12MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*15)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '15MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*18)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '18MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*21)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '21MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*24)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '24MonthWater'
    FROM (
    SELECT
        P.*
        ,RANK() OVER (PARTITION BY P.StateProducingUnitKey
                    ORDER BY P.OilRate_BBLPD DESC) as OilRank
        ,RANK() OVER (PARTITION BY P.StateProducingUnitKey
                    ORDER BY P.GasRate_MCFPD DESC) as GasRank
        ,RANK() OVER (PARTITION BY P.StateProducingUnitKey
                    ORDER BY P.WaterRate_BBLPD DESC) as WaterRank
    FROM(
        SELECT 
                    StateProducingUnitKey   
                    ,Max(ProductionDate) as ProductionDate
                    ,Max(DaysProducing) as DaysProducing
                    ,Sum(Oil) as Oil
                    ,Sum(Gas) as Gas
                    ,Sum(Water) as Water
                    ,Sum(CO2) as CO2
                    ,Sum(Sulfur) as Sulfur
                    ,Sum(Oil) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) CumOil
                    ,Sum(Gas) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) CumGas
                    ,Sum(Water) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) CumWater      
                    ,Sum(MAX(DaysProducing)) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) DaysOn
                    ,Sum(Oil) / MAX(DaysProducing) as OilRate_BBLPD
                    ,Sum(Gas) / MAX(DaysProducing) as GasRate_MCFPD
                    ,Sum(Water) / MAX(DaysProducing) as WaterRate_BBLPD
                    FROM PRODUCTION
    --                  WHERE CAST(StateProducingUnitKey as INTEGER) >= 5123449960000 AND CAST(StateProducingUnitKey as INTEGER) <= 5123450000000
                    GROUP BY StateProducingUnitKey, ProductionDate
                    ORDER BY  StateProducingUnitKey,ProductionDate ASC
                    ) P
    WHERE P.DaysOn <= 700
    ORDER BY  StateProducingUnitKey,ProductionDate ASC) PP
    -- WHERE CAST(PP.StateProducingUnitKey as INTEGER) >= 5123449960000 AND CAST(PP.StateProducingUnitKey as INTEGER) <= 5123450000000
    GROUP BY PP.StateProducingUnitKey
    """
    #quit()

    sqldb = "CO_3_2.1.sqlite"
    conn = sqlite3.connect(sqldb)
    c = conn.cursor()
    # c.execute(Query1)

    Well_df=pd.read_sql_query(Q1,conn)

    # Determine how to aggregate treatments
    c = conn.cursor()
    c.execute(Q2_Drop)
    c.execute(Q2.split(';')[0])
    # c.execute(Q2.split(';')[1)
    # df = pd.read_sql_query(Q2.split(';')[2],conn)
    df = pd.read_sql_query(Q2.split(';')[1],conn)
    df.loc[df.PastTreatmentDate!=None]
    df2 = pd.merge(df,pd.DataFrame(df).pivot_table(index=['StateWellKey'],aggfunc='size').rename('Count'),how='left',left_on='StateWellKey',right_on='StateWellKey')
    #(df.date2 - df.date1) / np.timedelta64(1, 'M')
    df2['TDelta']=(pd.to_datetime(df2.DATE1)-pd.to_datetime(df2.DATE2))/np.timedelta64(1,'D')/30.4
    x=df2.loc[(df2.Cum_Difference<50) | ((df2.TDelta <6) & (df2.Cum_Difference<1000)) | (df2.TDelta < 2)].sort_values(by='Cum_Difference')
    df2['TreatKey2']=0
    df2['TreatKey2']=((df2.Cum_Difference<50) | ((df2.TDelta <6) & (df2.Cum_Difference<1000)) | (df2.TDelta < 2))*1+df2.TreatKey2
    # For Stimulation order A-B-C
    # Dictionary receives stimulation B and returns A
    StimDictPrev = df2.loc[df2.TreatKey2>0].set_index('StateStimulationKey').PastStateStimulationKey.to_dict()
    # Dictionary receives stimulation B and returns C
    StimDict = df2.loc[df2.TreatKey2>0].set_index('PastStateStimulationKey').StateStimulationKey.to_dict()                          
    # StimDict to map and group
    # df2.loc[df2.TreatKey2>0].PastStateStimulationKey.map(StimDict).dropna()
    del df,df2,x                              

    # Get Stimulation Quantities
    STIM_df=pd.read_sql_query(Q3,conn)
    t_df = STIM_df.pivot(index = ['StateStimulationKey'], columns = ['Type'],values = ['TotalAmount','QC_CountUnits']).dropna(axis=1,how='all').dropna(axis=0,how='all')
    t_df.columns = ['_'.join(col).strip() for col in t_df.columns.values]

    STIM_df[['StateStimulationKey','TreatmentDate','StateWellKey']].drop_duplicates()

    STIM_df = pd.merge(t_df
        ,STIM_df[['StateStimulationKey','TreatmentDate','StateWellKey']].drop_duplicates()
        ,how = 'left'
        ,on = ['StateStimulationKey','StateStimulationKey'])
    del t_df
    i = 1
    STIM_df=STIM_df.drop(list(filter(re.compile('StateStimulationKey[0-9]').match, STIM_df.keys().to_list())),axis=1)    
    STIM_df['StateStimulationKey0']=STIM_df['StateStimulationKey']
    while STIM_df['StateStimulationKey'+str(i-1)].map(StimDict).dropna().shape[0]>0:
        STIM_df['StateStimulationKey'+str(i)] = STIM_df['StateStimulationKey'+str(i-1)]
        #STIM_df['StateStimulationKey'+str(i-1)].map(StimDict)
        STIM_df.loc[STIM_df['StateStimulationKey'+str(i-1)].map(StimDict).dropna().index,STIM_df.columns=='StateStimulationKey'+str(i)]=STIM_df['StateStimulationKey'+str(i-1)].map(StimDict).dropna()
        i+=1
    # Stimulation Dates Dictionary
    StimDates=STIM_df[['StateStimulationKey','TreatmentDate']].drop_duplicates().set_index('StateStimulationKey').astype('datetime64').TreatmentDate.to_dict()
    # Count of stimulations per well


    ##################################################################################################      UPDATE ME
        # UPDATE THIS TO TRACK INTERVAL OF COMPLETION ACTIVITY
        # AND ALSO FOLLOW THESE INTERVALS IN PRODUCTION
    ##################################################################################################
    ## Not sure, but I think this whole section has been superceded by lower sections
    ##if 1==1:
    ##    # Sum stimulation volumes where multiple stimulations in a short time
    ##    sumdf = STIM_df[(list(filter(re.compile('Total.*').match, STIM_df.keys().to_list())))+['StateStimulationKey'+str(i-1)]].groupby('StateStimulationKey'+str(i-1)).sum()    
    ##    # Max for QC Counts
    ##    maxdf = STIM_df[(list(filter(re.compile('QC_CountUnit.*').match, STIM_df.keys().to_list())))+['StateStimulationKey'+str(i-1)]].groupby('StateStimulationKey'+str(i-1)).max()
    ##    #Stimulation Summary Table
    ##    SS_df=sumdf.join(maxdf, on='StateStimulationKey'+str(i-1))
    ##    # Add Well API
    ##    SS_df.join(STIM_df[['StateStimulationKey'+str(i-1),'StateWellKey']].set_index('StateStimulationKey'+str(i-1)).drop_duplicates())
    ##    del x,maxdf,sumdf


    STIM_df['LastStimDate']=STIM_df['StateStimulationKey'+str(i-1)].map(StimDates)
    STIM_df[['StateWellKey','StateStimulationKey'+str(i-1),'LastStimDate']].drop_duplicates()
    ranks = STIM_df[['StateWellKey','StateStimulationKey'+str(i-1),'LastStimDate']].drop_duplicates() \
      .groupby(['StateWellKey'])['LastStimDate'] \
      .rank(ascending = True, method = 'first')-1
    ranks = ranks.fillna(0).astype('int64')

    RankDict = pd.Series(ranks.values.astype('int64')
                         ,index=STIM_df[['StateWellKey'
                         ,'StateStimulationKey'+str(i-1)
                         ,'LastStimDate']].drop_duplicates()['StateStimulationKey'+str(i-1)].values).to_dict()

    #ranks.name = 'StimRank'
    STIM_df['StimRank'] = STIM_df['StateStimulationKey'+str(i-1)].map(RankDict)
    STIM_df['StimRank'] = STIM_df['StimRank'].fillna(0)
    STIM_df['API14x']= (STIM_df.StimRank.astype('int64') + STIM_df.StateWellKey.astype('int64')).astype('str').str.zfill(14)

     #   STIM_df['StimDate0'] = STIM_df['StateStimulationKey'+str(i-1)].map(paststim).map(StimDates)
     #   STIM_df['StimDate1'] = STIM_df['StateStimulationKey'+str(i-1)].map(nextstim).map(StimDates)
     #   STIM_df.loc[STIM_df.LastStimDate == STIM_df.StimDate0,'StimDate0']=datetime.date(1900, 1, 1)
     #   STIM_df.loc[STIM_df.StimDate1 == STIM_df.StimDate1,'StimDate1']=datetime.datetime.now()

    # STIM_df.loc[STIM_df.StimRank>0]
    # STIM_df.loc[STIM_df.StimRank>0].StateWellKey
    # 0512340288000
    tbl = STIM_df[['StateWellKey','StateStimulationKey'+str(i-1),'LastStimDate','StimRank']]
    xbl = tbl.loc[tbl.StimRank == tbl.StimRank.max()].set_index('StateWellKey')

    dict1 = {};dict2 = {};
    for j in reversed(range(0,tbl.StimRank.max())):
        #suffix for rank n-1 values
        sfx = '_prev'
        x = pd.merge(tbl.loc[tbl.StimRank == j+1].set_index('StateWellKey')
             ,tbl.loc[tbl.StimRank == j].set_index('StateWellKey')
             ,how = 'inner'
             ,left_index = True
             ,right_index = True
             ,suffixes =('',sfx))

        dict1.update(x[['StateStimulationKey'+str(i-1),'StateStimulationKey'+str(i-1)+sfx]].set_index('StateStimulationKey'+str(i-1))['StateStimulationKey'+str(i-1)+sfx].to_dict())
        dict2.update(x[['StateStimulationKey'+str(i-1)+sfx,'StateStimulationKey'+str(i-1)]].set_index('StateStimulationKey'+str(i-1)+sfx)['StateStimulationKey'+str(i-1)].to_dict())
        # dict2.update(dict1)

    STIM_df['StimDate0'] = STIM_df['StateStimulationKey'+str(i-1)].map(dict2).map(StimDates).fillna(datetime.date(1900, 1, 1))
    STIM_df['StimDate1'] = STIM_df['StateStimulationKey'+str(i-1)].map(dict1).map(StimDates).fillna(datetime.datetime.now())
    STIM_df.loc[STIM_df.LastStimDate == STIM_df.StimDate0,'StimDate0']=datetime.date(1900, 1, 1)
    STIM_df.loc[STIM_df.StimDate1 == STIM_df.StimDate1,'StimDate1']=datetime.datetime.now()

    # 3,6,9,12,15,18,21,24 month cum
    cumdf = pd.read_sql_query(Q4,conn)
    cumdf = cumdf.set_index('StateProducingUnitKey',drop=False)
    #cumdf['StateProducingUnitKey'] = cumdf.index

    # adds empty columns 
    #cumdf = cumdf.reindex(cumdf.columns.tolist() + STIM_df.keys().tolist(), axis=1)
    #cumdf[STIM_df.keys()]=np.nan

    # For speed
    # 1) Assign all wells to 0th conpletion
    # 2) For each well with multiple completions, get assignments
    well_list = cumdf.index.intersection(STIM_df.loc[STIM_df.StimRank>0]['StateWellKey'].unique())

    cumdf = pd.merge(cumdf,STIM_df.drop_duplicates(['StateWellKey']).set_index('StateWellKey'),how='outer',left_index = True, right_index=True)
    STIM_df.StimDate0 = pd.to_datetime(STIM_df.StimDate0)
    cumdf.PeakOil_Date = pd.to_datetime(cumdf.PeakOil_Date)
    for well in well_list:
        x = STIM_df.loc[(STIM_df.StateWellKey == well) \
            & (STIM_df.StimDate0 <= cumdf.loc[well,'PeakOil_Date'])]
        if x.shape[0] ==0:
            continue
        elif x.shape[0] >1:
            x=x.sort_values(by='StimDate0',ascending=False).iloc[0,:]
        else:
            pass
        cumdf.loc[well,STIM_df.keys()]=x.squeeze()

    #StimPivot = STIM_df.pivot_table(values = 'LastStimDate',index = 'StateWellKey',columns = 'StimRank',aggfunc='max')                                                                                                                                        
    #StimPivot.loc[StimPivot.index == '05125121260000'].dropna(axis=1)

    # Get Perforation Detail
    Perf_df = pd.read_sql_query("""SELECT * FROM WellPerf""",conn)
    cumdf=pd.merge(cumdf,Perf_df.drop_duplicates('StateWellKey').set_index('StateWellKey'),how='outer',left_index = True, right_index=True)
    #Perf Interval
    cumdf['PerfInterval'] = cumdf.Bottom-cumdf.Top
    # Frac Per Ft
    cumdf['TotalFluid'] = cumdf['TotalAmount_FRESH WATER']+cumdf['TotalAmount_RECYCLED WATER']
    cumdf['Stimulation_FluidBBLPerFt'] = cumdf['TotalFluid']/cumdf['PerfInterval']
    cumdf['Stimulation_ProppantLBSPerFt'] = cumdf['TotalAmount_PROPPANT']/cumdf['PerfInterval']
    cumdf['Stimulation_ProppantLBSPerBBL'] = cumdf['TotalAmount_PROPPANT']/cumdf['TotalFluid']
    conn.close()

    # MERGE WITH WELL HEADERS
    with sqlite3.connect('CO_3_2.1.sqlite') as conn:
        Well_df=pd.read_sql_query("""SELECT * FROM WELL""",conn)
    ULT = pd.merge(cumdf,Well_df.set_index('StateWellKey'), left_index=True, right_index=True,how='outer')
    for k in ULT.keys():
        if 'DATE' in k.upper():
            ULT[k] = pd.to_datetime(ULT['FirstCompDate'])

    ULT.to_json('SQL_WELL_SUMMARY.JSON')
    ULT.to_parquet('SQL_WELL_SUMMARY.PARQUET')

    # dtypes for SQLITE3
    pd_sql_types={'object':'TEXT',
                  'int64':'INTEGER',
                  'float64':'FLOAT',
                  'bool':'TEXT',
                  'datetime64':'TEXT',
                  'datetime64[ns]':'TEXT',
                  'timedelta[ns]':'REAL',
                  'category':'TEXT'
        }

    #dtypes for SQLALCHEMY
    pd_sql_types={'object':sqlalchemy.types.Text,
                  'int64':sqlalchemy.types.BigInteger,
                  'float64':sqlalchemy.types.Float,
                  'bool':sqlalchemy.types.Boolean,
                  'datetime64':sqlalchemy.types.Date,
                  'datetime64[ns]':sqlalchemy.types.Date,
                  'timedelta[ns]':sqlalchemy.types.Float,
                  'category':sqlalchemy.types.Text
        }

    df_typemap = ULT.dtypes.astype('str').map(pd_sql_types).to_dict()
    if SAVE:               
        engine = sqlalchemy.create_engine(f'sqlite:///{DB}')
        with engine.begin() as connection:
            ULT.to_sql(TABLE_NAME,connection,
                      if_exists='replace',
                      index=False,
                      dtype=df_typemap)
    
def SUMMARIZE_PROD_DATA(pdf, ADD_RATIOS = False):
    #pathname = path.dirname(argv[0])
    #adir = path.abspath(pathname)
    adir = getcwd()

    pdf[['PROD_DAYS','OIL_RATE','GAS_RATE','WTR_RATE','PROD_DAYS','CUMOIL','CUMGAS','CUMWTR','TMB_OIL','TMB_GAS','TMB_WTR','GOR','OWR','WOR','OWC','WOC']] = np.nan
    
    OUTPUT=pd.DataFrame(columns=['UWI','UWI10',
                                 'BTU_MEAN','BTU_STD'
                                 ,'API_MEAN','API_STD'
                                 ,'Peak_Oil_Date','Peak_Oil_Days','Peak_Oil_CumOil','Peak_Oil_CumGas','Peak_Oil_CumWtr'
                                 ,'Peak_Gas_Date','Peak_Gas_Days','Peak_Gas_CumOil','Peak_Gas_CumGas','Peak_Gas_CumWtr'
                                 ,'OWR_PrePeakOil','OWR_PostPeakGas'
                                 ,'GOR_PrePeakOil','GOR_PeakGas','GOR_PostPeakGOR'
                                 ,'WOC_PostPeakOil','WOC_PostPeakGas'
                                 ,'GOR_Final','OWC_Final'
                                 ,'Month1'
                                 ,'GOR_MO2_4','GOR_MO5_7','GOR_MO11_13','GOR_MO23_25','GOR_MO35_37','GOR_MO47_49'
                                 ,'OWR_MO2_4','OWR_MO5_7','OWR_MO11_13','OWR_MO23_25','OWR_MO35_37','OWR_MO47_49'
                                 ,'Production_Formation'])
    
    MonthArray = np.arange(3,49,3)
    for i in MonthArray:
        OUTPUT['CumOil_Mo'+str(i)] = np.nan
        OUTPUT['CumGas_Mo'+str(i)] = np.nan
        OUTPUT['CumWtr_Mo'+str(i)] = np.nan
    
    if 'UWI10' in pdf.keys():
        UWIKEY = 'UWI10'
    else:
        UWIKEY = Find_Str_Locs(pdf,'UWI|API')
        try:
            UWIKEY = pdf[UWIKEY].map(lambda x:WELLAPI(x).API2INT(10)>0).sum(axis=0).sort_values(axis=0, ascending = False).keys()[0]
        except:
            print('NO UWI COLUMN!')
            return None
           
    try: 
        DATE     = pdf.iloc[:,pdf.keys().str.contains('.*FIRST.*MONTH.*|.*REPORT.*DATE.*', regex=True, case=False,na=False)].keys()[0]
        DAYSON   = pdf.iloc[0,pdf.keys().str.contains('.*DAYS.*PROD.*|.*PROD.*DAYS.*', regex=True, case=False,na=False)].keys()[0]
        OIL      = pdf.iloc[0,pdf.keys().str.contains('.*OIL.*PROD.*', regex=True, case=False,na=False)].keys()[0]
        GAS      = pdf.iloc[0,pdf.keys().str.contains('.*GAS.*PROD.*', regex=True, case=False,na=False)].keys()[0]
        WTR      = pdf.iloc[0,pdf.keys().str.contains('.*WATER.*VOLUME.*', regex=True, case=False,na=False)].keys()[0]
        API      = pdf.iloc[0,pdf.keys().str.contains('.*OIL.*GRAVITY.*', regex=True, case=False,na=False)].keys()[0]
        BTU      = pdf.iloc[0,pdf.keys().str.contains('.*GAS.*BTU.*', regex=True, case=False,na=False)].keys()[0]
        FM       = pdf.iloc[0,pdf.keys().str.contains('.*Formation.*|.*POOL.*', regex=True, case=False,na=False)].keys()[0]
        #SEQ      = pdf.iloc[0,pdf.keys().str.contains('.*SEQUENCE.*', regex=True, case=False,na=False)].keys()[0]

    except:
        print(f'Cannot parse tables')
        return None
    
    pdf[DATE] = pd.to_datetime(pdf[DATE]).dt.date
           
    for UWI in pdf[UWIKEY].unique():
        #UWI
        mB = pdf[UWIKEY] == UWI
        m = pdf.index[mB]
           
        if pdf.loc[m,[OIL,GAS,WTR]].dropna(how='any').shape[0]==0:
           #print('NO PRODUCTION')
           continue
        OUTPUT.at[UWI,'UWI'] = UWI
        OUTPUT.at[UWI,'UWI10'] = WELLAPI(UWI).API2INT(10)

        pdf.loc[m,'PROD_DAYS'] = pdf.loc[m,DAYSON].cumsum()
        
        pdf.sort_values(by=DATE, ascending = True, inplace =True)

        PRODOIL = pdf.loc[m,OIL].dropna().index
        PRODGAS = pdf.loc[m,GAS].dropna().index
        PRODWTR = pdf.loc[m,WTR].dropna().index

        pdf.loc[m,'OIL_RATE'] = pdf.loc[m,OIL]/pdf.loc[m,DAYSON]
        pdf.loc[m,'GAS_RATE'] = pdf.loc[m,GAS]/pdf.loc[m,DAYSON]
        pdf.loc[m,'WTR_RATE'] = pdf.loc[m,WTR]/pdf.loc[m,DAYSON]
        pdf.loc[m,'PROD_DAYS'] = pdf.loc[m,DAYSON].cumsum()
        
        pdf.loc[m,'CUMOIL'] = pdf.loc[m,OIL].cumsum()
        pdf.loc[m,'CUMGAS'] = pdf.loc[m,GAS].cumsum()
        pdf.loc[m,'CUMWTR'] = pdf.loc[m,WTR].cumsum()

        #pdf[['TMB_OIL','TMB_GAS','TMB_WTR']] = np.nan
        
        pdf.loc[PRODOIL,'TMB_OIL'] = pdf.loc[PRODOIL,'CUMOIL'] / pdf.loc[PRODOIL,OIL]
        pdf.loc[PRODGAS,'TMB_GAS'] = pdf.loc[PRODGAS,'CUMGAS'] / pdf.loc[PRODGAS,GAS]
        pdf.loc[PRODWTR,'TMB_WTR'] = pdf.loc[PRODWTR,'CUMWTR'] / pdf.loc[PRODWTR,WTR]      

        #pdf[['GOR','OWR','WOR','OWC','WOC']] = np.nan
        pdf.loc[PRODOIL,'GOR'] = pdf.loc[PRODOIL,GAS]*1000/pdf.loc[PRODOIL,OIL]
        pdf.loc[PRODWTR,'OWR'] = pdf.loc[PRODWTR,OIL]/pdf.loc[PRODWTR,WTR]
        pdf.loc[PRODOIL,'WOR'] = pdf.loc[PRODOIL,WTR]/pdf.loc[PRODOIL,OIL]
        
        m2 = PRODOIL.join(PRODWTR,how='outer')
        pdf.loc[m2,'OWC'] = pdf.loc[m2,OIL]/(pdf.loc[m2,WTR]+pdf.loc[m2,OIL])
        pdf.loc[m2,'WOC'] = pdf.loc[m2,WTR]/(pdf.loc[m2,WTR]+pdf.loc[m2,OIL])
        
        if pdf[[API]].dropna(how='any').shape[0]>3:
            OUTPUT.at[UWI,'API_MEAN']         = pdf.loc[m,API].astype('float').describe()[1]
            OUTPUT.at[UWI,'API_STD']          = pdf.loc[m,API].astype('float').describe()[2]
        if pdf[[BTU]].dropna(how='any').shape[0]>3:
            OUTPUT.at[UWI,'BTU_MEAN']         = pdf.loc[m,BTU].astype('float').describe()[1]
            OUTPUT.at[UWI,'BTU_STD']          = pdf.loc[m,BTU].astype('float').describe()[2]
        if pdf[[OIL,GAS]].dropna(how='any').shape[0]>3:
            OUTPUT.at[UWI,'Peak_Oil_Date']   = pdf.loc[m,DATE][pdf.loc[m,OIL].idxmax()]
            OUTPUT.at[UWI,'Peak_Oil_Days']   = pdf.loc[m,'PROD_DAYS'][pdf.loc[m,OIL].idxmax()]
            OUTPUT.at[UWI,'Peak_Oil_CumOil'] = pdf.loc[m,OIL][0:pdf.loc[m,OIL].idxmax()].sum()
            OUTPUT.at[UWI,'Peak_Oil_CumGas'] = pdf.loc[m,GAS][0:pdf.loc[m,OIL].idxmax()].sum()

            OUTPUT.at[UWI,'Peak_Gas_Date']   = pdf.loc[m,DATE][pdf.loc[m,GAS].idxmax()]
            OUTPUT.at[UWI,'Peak_Gas_Days']   = pdf.loc[m,'PROD_DAYS'][pdf.loc[m,GAS].idxmax()]
            OUTPUT.at[UWI,'Peak_Gas_CumOil'] = pdf.loc[m,OIL][0:pdf.loc[m,GAS].idxmax()].sum()
            OUTPUT.at[UWI,'Peak_Gas_CumGas'] = pdf.loc[m,GAS][0:pdf.loc[m,GAS].idxmax()].sum()

            #PREPEAKOIL  = pdf.index[mB][(pdf.loc[m,'PROD_DAYS']-pdf.loc[m,'PROD_DAYS'][pdf.loc[m,OIL].idxmax()]).between(-100,0)]
            PREPEAKOIL  = pdf.loc[m].index[(pdf.loc[m,'PROD_DAYS'] - pdf.loc[pdf.loc[m,OIL].idxmax(),'PROD_DAYS']).between(-200,0)]
            PREPEAKGAS  = pdf.loc[m].index[(pdf.loc[m,'PROD_DAYS'] - pdf.loc[pdf.loc[m,GAS].idxmax(),'PROD_DAYS']).between(-200,0)]
                      
            #POSTPEAKOIL = pdf.loc[(pdf.loc[m,'PROD_DAYS'][pdf.loc[m,OIL].idxmax()]-pdf.loc[m,'PROD_DAYS']).between(0,100),:].index
            POSTPEAKOIL  = pdf.loc[m].index[(pdf.loc[m,'PROD_DAYS'] - pdf.loc[pdf.loc[m,OIL].idxmax(),'PROD_DAYS']).between(0,100)]
            
            #POSTPEAKGAS = pdf.loc[(pdf.loc[m,'PROD_DAYS'][pdf.loc[m,GAS].idxmax()]-pdf.loc[m,'PROD_DAYS']).between(0,100),:].index
            POSTPEAKGAS  = pdf.loc[m].index[(pdf.loc[m,'PROD_DAYS'] - pdf.loc[pdf.loc[m,GAS].idxmax(),'PROD_DAYS']).between(0,100)]
            
            #PEAKGAS = pdf.loc[(pdf.loc[m,'PROD_DAYS'][pdf.loc[m,GAS].idxmax()]-pdf.loc[m,'PROD_DAYS']).between(-50,50),:].index
            PEAKGAS = pdf.loc[m].index[(pdf.loc[m,'PROD_DAYS'] - pdf.loc[pdf.loc[m,GAS].idxmax(),'PROD_DAYS']).between(-50,50)]
           
            LATEWATER = pdf.loc[m].index[pdf.loc[m,'TMB_WTR']>500]
            LATEGAS = pdf.loc[m].index[pdf.loc[m,'TMB_GAS']>30]
            
            if pdf.loc[PREPEAKOIL,OIL].sum()>0:
                OUTPUT.at[UWI,'GOR_PrePeakOil']  = pdf.loc[PREPEAKOIL,GAS].sum() * 1000 / pdf.loc[PREPEAKOIL,OIL].sum()
            if pdf.loc[PEAKGAS,OIL].sum()>0:
                OUTPUT.at[UWI,'GOR_PeakGas']     = pdf.loc[PEAKGAS,GAS].sum() * 1000 / pdf.loc[PEAKGAS,OIL].sum()
                               
            if len(PRODOIL.intersection(PRODGAS).intersection(PRODWTR)) >3 : 
                if pdf.loc[PREPEAKOIL,WTR].sum()>0:
                    OUTPUT.at[UWI,'OWR_PrePeakOil']  = pdf.loc[PREPEAKOIL,OIL].sum()/pdf.loc[PREPEAKOIL,WTR].sum()
                if pdf.loc[POSTPEAKGAS,WTR].sum() >0:
                    OUTPUT.at[UWI,'OWR_PostPeakGas'] = pdf.loc[POSTPEAKGAS,OIL].sum()/pdf.loc[POSTPEAKGAS,WTR].sum()     
                from scipy.special import logsumexp
                OUTPUT.at[UWI,'WOC_PostPeakOil'] = pdf.loc[POSTPEAKOIL,WTR].sum() / (pdf.loc[POSTPEAKOIL,WTR].sum()+pdf.loc[POSTPEAKOIL,OIL].sum())                          
                
                OUTPUT.at[UWI,'WOC_PostPeakGas'] = pdf.loc[POSTPEAKGAS,WTR].sum() / (pdf.loc[POSTPEAKGAS,WTR].sum()+pdf.loc[POSTPEAKGAS,OIL].sum())        
                OUTPUT.at[UWI,'Peak_Oil_CumWtr'] = pdf.loc[m,WTR][0:pdf.loc[m,OIL].idxmax()].sum()
                OUTPUT.at[UWI,'Peak_Gas_CumWtr'] = pdf.loc[m,WTR][0:pdf.loc[m,GAS].idxmax()].sum()
              
                if len(LATEGAS)>3:
                    OUTPUT.at[UWI,'GOR_Final'] = pdf.loc[LATEGAS, GAS].sum() / pdf.loc[LATEGAS, OIL].sum() * 1000
                if len(LATEWATER)>3:
                    OUTPUT.at[UWI,'OWC_Final'] =  pdf.loc[LATEWATER, OIL].sum() / (pdf.loc[LATEWATER, OIL].sum()+pdf.loc[LATEWATER, WTR].sum())

            if len(pdf[DATE].dropna())>10:
                MONTH1 = pdf.loc[mB & (pdf[DAYSON]>14) & (pdf[OIL]>0),DATE].min()
                OUTPUT.at[UWI,'Month1'] = MONTH1

                if not isinstance(MONTH1,float):
                    pdf['EM_PRODMONTH'] = (pd.to_datetime(pdf.loc[m,DATE]).dt.year - MONTH1.year)*12+(pd.to_datetime(pdf.loc[m,DATE]).dt.month - MONTH1.month)+1

                    for i in MonthArray:
                        if max(pdf.loc[m,'EM_PRODMONTH']) >= i:
                            i_dwn = i-1
                            i_up = i+1
                            OUTPUT.at[UWI,'CumOil_Mo'+str(i)] = pdf.loc[mB & (pdf['EM_PRODMONTH']<=i),OIL].sum()
                            OUTPUT.at[UWI,'CumGas_Mo'+str(i)] = pdf.loc[mB & (pdf['EM_PRODMONTH']<=i),GAS].sum()
                            OUTPUT.at[UWI,'CumWtr_Mo'+str(i)] = pdf.loc[mB & (pdf['EM_PRODMONTH']<=i),WTR].sum()
                                    
                            if pdf.loc[mB & (pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum() > 0:
                                OUTPUT.at[UWI,'GOR_MO'+str(i_dwn)+'_'+str(i_up)]  = pdf.loc[mB & (pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum()
                            if (pdf.loc[mB & (pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),WTR].sum()) >=1:   
                                OUTPUT.at[UWI,'OWC_MO'+str(i)] = pdf.loc[mB & (pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),WTR].sum())
                            if pdf.loc[mB & (pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),WTR].sum() > 0: 
                                OUTPUT.at[UWI,'OWR_MO'+str(i_dwn)+'_'+str(i_up)]  = pdf.loc[mB & (pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),WTR].sum()
        OUTPUT.at[UWI,'Production_Formation'] = '_'.join(pdf.loc[m,FM].unique())
        #pdf.loc[m,'UWI'] = UWI
           
    if ADD_RATIOS:
        return OUTPUT, pdf
    else:
        return OUTPUT

def SUMMARIZE_PROD_DATA2(ppdf, ADD_RATIOS = False):
    warnings.filterwarnings('ignore')
    adir = getcwd()

    ppdf[['PROD_DAYS','OIL_RATE','GAS_RATE','WTR_RATE','PROD_DAYS','CUMOIL','CUMGAS','CUMWTR','TMB_OIL','TMB_GAS','TMB_WTR','GOR','OWR','WOR','OWC','WOC']] = np.nan
    
    OUTPUT=pd.DataFrame(columns=['UWI','UWI10','FIRST_PRODUCTION'
                                 ,'BTU_MEAN','BTU_STD'
                                 ,'API_MEAN','API_STD'
                                 ,'Peak_Oil_Date','Peak_Oil_Days','Peak_Oil_CumOil','Peak_Oil_CumGas','Peak_Oil_CumWtr'
                                 ,'Peak_Gas_Date','Peak_Gas_Days','Peak_Gas_CumOil','Peak_Gas_CumGas','Peak_Gas_CumWtr'
                                 ,'OWR_PrePeakOil','OWR_PostPeakGas'
                                 ,'GOR_PrePeakOil','GOR_PeakGas','GOR_PostPeakGOR'
                                 ,'WOC_PostPeakOil','WOC_PostPeakGas'
                                 ,'GOR_Final','OWC_Final'
                                 ,'Month1'
                                 ,'GOR_MO2_4','GOR_MO5_7','GOR_MO11_13','GOR_MO23_25','GOR_MO35_37','GOR_MO47_49'
                                 ,'OWR_MO2_4','OWR_MO5_7','OWR_MO11_13','OWR_MO23_25','OWR_MO35_37','OWR_MO47_49'
                                 ,'Production_Formation'])
    
    MonthArray = np.arange(3,49,3)
    for i in MonthArray:
        OUTPUT['CumOil_Mo'+str(i)] = np.nan
        OUTPUT['CumGas_Mo'+str(i)] = np.nan
        OUTPUT['CumWtr_Mo'+str(i)] = np.nan
    
    if 'UWI10' in ppdf.keys():
        UWIKEY = 'UWI10'
    else:
        #UWIKEY = Find_Str_Locs(ppdf,'UWI|API')
        UWIKEY = GetKey(ppdf,r'UWI|API')
        try:
            UWIKEY = ppdf[UWIKEY].applymap(lambda x:WELLAPI(x).API2INT(10)>0).sum(axis=0).sort_values(axis=0, ascending = False).keys()[0]
        except:
            print('NO UWI COLUMN!')
            return None
           
    try: 
        DATE     = ppdf.iloc[:,ppdf.keys().str.contains('.*FIRST.*MONTH.*', regex=True, case=False,na=False)].keys()[0]
        DAYSON   = ppdf.iloc[0,ppdf.keys().str.contains('.*DAYS.*(ON|PROD).*', regex=True, case=False,na=False)].keys()[0]
        OIL      = ppdf.iloc[0,ppdf.keys().str.contains('.*OIL.*(PROD|VOL).*', regex=True, case=False,na=False)].keys()[0]
        GAS      = ppdf.iloc[0,ppdf.keys().str.contains('.*GAS.*(PROD|VOL).*', regex=True, case=False,na=False)].keys()[0]
        WTR      = ppdf.iloc[0,ppdf.keys().str.contains('.*WATER.*(PROD|VOL).*', regex=True, case=False,na=False)].keys()[0]
        API      = ppdf.iloc[0,ppdf.keys().str.contains('.*OIL.*GRAVITY.*', regex=True, case=False,na=False)].keys()[0]
        BTU      = ppdf.iloc[0,ppdf.keys().str.contains('.*GAS.*(GRAVITY|BTU).*', regex=True, case=False,na=False)].keys()[0]
        FM       = ppdf.iloc[0,ppdf.keys().str.contains('.*Formation.*', regex=True, case=False,na=False)].keys()[0]
        #SEQ      = pdf.iloc[0,pdf.keys().str.contains('.*SEQUENCE.*', regex=True, case=False,na=False)].keys()[0]

    except:
        print(f'Cannot parse tables')
        ERROR = 1
        return None

    for k in [DAYSON,OIL,GAS,WTR,API,BTU]:
        ppdf[k] = pd.to_numeric(ppdf[k])
           
    ppdf[DATE] = pd.to_datetime(ppdf[DATE]).dt.date
    ppdf.sort_values(by = DATE, ascending = True, inplace = True, ignore_index =True)

    PRODOIL = ppdf[OIL].dropna().index
    PRODGAS = ppdf[GAS].dropna().index
    PRODWTR = ppdf[WTR].dropna().index

    # CUM VALUES
    ppdf['CUMOIL'] = ppdf[[UWIKEY,OIL]].groupby([UWIKEY])[OIL].transform('cumsum', skipna = True)
    ppdf['CUMGAS'] = ppdf[[UWIKEY,GAS]].groupby([UWIKEY])[GAS].transform('cumsum', skipna = True)
    ppdf['CUMWTR'] = ppdf[[UWIKEY,WTR]].groupby([UWIKEY])[WTR].transform('cumsum', skipna = True)

    #MASS BALANCE TIME
    ppdf.loc[PRODOIL,'TMB_OIL'] = ppdf.loc[PRODOIL,'CUMOIL'] / ppdf.loc[PRODOIL,OIL]
    ppdf.loc[PRODGAS,'TMB_GAS'] = ppdf.loc[PRODGAS,'CUMGAS'] / ppdf.loc[PRODGAS,GAS]
    ppdf.loc[PRODWTR,'TMB_WTR'] = ppdf.loc[PRODWTR,'CUMWTR'] / ppdf.loc[PRODWTR,WTR]      
    
    # RATES
    ppdf['OIL_RATE'] = ppdf[OIL]/ppdf[DAYSON]
    ppdf['GAS_RATE'] = ppdf[GAS]/ppdf[DAYSON]
    ppdf['WTR_RATE'] = ppdf[WTR]/ppdf[DAYSON]
    ppdf['PROD_DAYS'] = ppdf[[UWIKEY ,DAYSON]].groupby([UWIKEY]).cumsum()

    # NORM PARAMS
    ppdf[['NORM_OIL','NORM_GAS','NORM_WTR']] = np.nan
    ppdf['NORM_OIL'] = ppdf[OIL]/ppdf.groupby([UWIKEY])[OIL].cummax(skipna=True)
    ppdf['NORM_GAS'] = ppdf[GAS]/ppdf.groupby([UWIKEY])[GAS].cummax(skipna=True)
    ppdf['NORM_WTR'] = ppdf[WTR]/ppdf.groupby([UWIKEY])[WTR].cummax(skipna=True)

    #pdf[['GOR','OWR','WOR','OWC','WOC']] = np.nan
    ppdf.loc[PRODOIL,'GOR'] = ppdf.loc[PRODOIL,GAS]*1000/ppdf.loc[PRODOIL,OIL]
    ppdf.loc[PRODWTR,'OWR'] = ppdf.loc[PRODWTR,OIL]/ppdf.loc[PRODWTR,WTR]
    ppdf.loc[PRODOIL,'WOR'] = ppdf.loc[PRODOIL,WTR]/ppdf.loc[PRODOIL,OIL]

    m2 = PRODOIL.join(PRODWTR,how='outer')
    ppdf.loc[m2,'OWC'] = ppdf.loc[m2,OIL]/(ppdf.loc[m2,WTR]+ppdf.loc[m2,OIL])
    ppdf.loc[m2,'WOC'] = ppdf.loc[m2,WTR]/(ppdf.loc[m2,WTR]+ppdf.loc[m2,OIL])

    for UWI in ppdf[UWIKEY].unique():
        pdf = ppdf.loc[ppdf[UWIKEY] == UWI,:].copy()
        pdf.sort_values(by=  DATE, ascending = True).reset_index(drop=True, inplace = True)
           
        if pdf[[OIL,GAS,WTR]].dropna(how='any').shape[0]<=6:
           #print('NO PRODUCTION')
           continue
        OUTPUT.at[UWI,'UWI'] = UWI
        OUTPUT.at[UWI,'UWI10'] = WELLAPI(UWI).API2INT(10)
           
        OUTPUT.at[UWI,'FIRST_PRODUCTION'] = pdf[[OIL,GAS,WTR,DATE]].dropna(how='all')[DATE].min()

        pdf['PROD_DAYS'] = pdf[DAYSON].cumsum()
        
        pdf.sort_values(by=DATE, ascending = True, inplace =True)

        PRODOIL = pdf[OIL].dropna().index
        PRODGAS = pdf[GAS].dropna().index
        PRODWTR = pdf[WTR].dropna().index

        m2 = PRODOIL.join(PRODWTR,how='outer')
        
        if pdf[[API]].dropna(how='any').shape[0]>3:
            OUTPUT.at[UWI,'API_MEAN']         = pdf[API].astype('float').describe()[1]
            OUTPUT.at[UWI,'API_STD']          = pdf[API].astype('float').describe()[2]
        if pdf[[BTU]].dropna(how='any').shape[0]>3:
            OUTPUT.at[UWI,'BTU_MEAN']         = pdf[BTU].astype('float').describe()[1]
            OUTPUT.at[UWI,'BTU_STD']          = pdf[BTU].astype('float').describe()[2]
        if pdf[[OIL,GAS]].dropna(how='any').shape[0]>3:
            OUTPUT.at[UWI,'Peak_Oil_Date']   = pdf[DATE][pdf[OIL].idxmax()]
            OUTPUT.at[UWI,'Peak_Oil_Days']   = pdf['PROD_DAYS'][pdf[OIL].idxmax()]
            OUTPUT.at[UWI,'Peak_Oil_CumOil'] = pdf[OIL][0:pdf[OIL].idxmax()].sum()
            OUTPUT.at[UWI,'Peak_Oil_CumGas'] = pdf[GAS][0:pdf[OIL].idxmax()].sum()

            OUTPUT.at[UWI,'Peak_Gas_Date']   = pdf[DATE][pdf[GAS].idxmax()]
            OUTPUT.at[UWI,'Peak_Gas_Days']   = pdf['PROD_DAYS'][pdf[GAS].idxmax()]
            OUTPUT.at[UWI,'Peak_Gas_CumOil'] = pdf[OIL][0:pdf[GAS].idxmax()].sum()
            OUTPUT.at[UWI,'Peak_Gas_CumGas'] = pdf[GAS][0:pdf[GAS].idxmax()].sum()

            #PREPEAKOIL  = pdf.index[mB][(pdf.loc[m,'PROD_DAYS']-pdf.loc[m,'PROD_DAYS'][pdf.loc[m,OIL].idxmax()]).between(-100,0)]
            PREPEAKOIL  = pdf.index[(pdf['PROD_DAYS'] - pdf.loc[pdf[OIL].idxmax(),'PROD_DAYS']).between(-200,0)]
           
            #POSTPEAKOIL = pdf.loc[(pdf.loc[m,'PROD_DAYS'][pdf.loc[m,OIL].idxmax()]-pdf.loc[m,'PROD_DAYS']).between(0,100),:].index
            POSTPEAKOIL  = pdf.index[(pdf['PROD_DAYS'] - pdf.loc[pdf[OIL].idxmax(),'PROD_DAYS']).between(0,100)]
            
            #POSTPEAKGAS = pdf.loc[(pdf.loc[m,'PROD_DAYS'][pdf.loc[m,GAS].idxmax()]-pdf.loc[m,'PROD_DAYS']).between(0,100),:].index
            POSTPEAKGAS  = pdf.index[(pdf['PROD_DAYS'] - pdf.loc[pdf[GAS].idxmax(),'PROD_DAYS']).between(0,100)]
            
            #PEAKGAS = pdf.loc[(pdf.loc[m,'PROD_DAYS'][pdf.loc[m,GAS].idxmax()]-pdf.loc[m,'PROD_DAYS']).between(-50,50),:].index
            PEAKGAS = pdf.index[(pdf['PROD_DAYS'] - pdf.loc[pdf[GAS].idxmax(),'PROD_DAYS']).between(-50,50)]
           
            LATEWATER = pdf.index[pdf['TMB_WTR']>500]
            LATEGAS = pdf.index[pdf['TMB_GAS']>30]
            
            if pdf.loc[PREPEAKOIL,OIL].sum()>0:
                OUTPUT.at[UWI,'GOR_PrePeakOil']  = pdf.loc[PREPEAKOIL,GAS].sum() * 1000 / pdf.loc[PREPEAKOIL,OIL].sum()
            if pdf.loc[PEAKGAS,OIL].sum()>0:
                OUTPUT.at[UWI,'GOR_PeakGas']     = pdf.loc[PEAKGAS,GAS].sum() * 1000 / pdf.loc[PEAKGAS,OIL].sum()
                               
            if len(PRODOIL.intersection(PRODGAS).intersection(PRODWTR)) >3 : 
                if pdf.loc[PREPEAKOIL,WTR].sum()>0:
                    OUTPUT.at[UWI,'OWR_PrePeakOil']  = np.exp(logsumexp(pdf.loc[PREPEAKOIL,OIL])-logsumexp(pdf.loc[PREPEAKOIL,WTR]))
                    #OUTPUT.at[UWI,'OWR_PrePeakOil']  = pdf.loc[PREPEAKOIL,OIL].sum()/pdf.loc[PREPEAKOIL,WTR].sum()
                if pdf.loc[POSTPEAKGAS,WTR].sum() >0:
                    OUTPUT.at[UWI,'OWR_PostPeakGas']  = np.exp(logsumexp(pdf.loc[POSTPEAKGAS,OIL])-logsumexp(pdf.loc[POSTPEAKGAS,WTR]))
                    #OUTPUT.at[UWI,'OWR_PostPeakGas'] = pdf.loc[POSTPEAKGAS,OIL].sum()/pdf.loc[POSTPEAKGAS,WTR].sum()   
                if pdf.loc[POSTPEAKOIL,[WTR,OIL]].dropna(how = 'any').shape[0]>2:
                    OUTPUT.at[UWI,'WOC_PostPeakOil'] = np.exp(logsumexp(pdf.loc[POSTPEAKOIL,WTR]) - logsumexp(pdf.loc[POSTPEAKOIL,WTR].fillna(0)+pdf.loc[POSTPEAKOIL,OIL].fillna(0)))                                                                          
                    #OUTPUT.at[UWI,'WOC_PostPeakOil'] = pdf.loc[POSTPEAKOIL,WTR].sum() / (pdf.loc[POSTPEAKOIL,WTR].sum()+pdf.loc[POSTPEAKOIL,OIL].sum())
                if pdf.loc[POSTPEAKGAS,[WTR,OIL]].dropna(how = 'any').shape[0]>2:
                    OUTPUT.at[UWI,'WOC_PostPeakGas'] = np.exp(logsumexp(pdf.loc[POSTPEAKGAS,WTR]) - logsumexp(pdf.loc[POSTPEAKGAS,WTR].fillna(0)+pdf.loc[POSTPEAKGAS,OIL].fillna(0)))                                                               
                    #OUTPUT.at[UWI,'WOC_PostPeakGas'] = pdf.loc[POSTPEAKGAS,WTR].sum() / (pdf.loc[POSTPEAKGAS,WTR].sum()+pdf.loc[POSTPEAKGAS,OIL].sum())       
                       
                OUTPUT.at[UWI,'Peak_Oil_CumWtr'] = pdf[WTR][0:pdf[OIL].idxmax()].sum()
                OUTPUT.at[UWI,'Peak_Gas_CumWtr'] = pdf[WTR][0:pdf[GAS].idxmax()].sum()
              
                if len(LATEGAS)>3:
                    OUTPUT.at[UWI,'GOR_Final'] = pdf.loc[LATEGAS, GAS].sum() / pdf.loc[LATEGAS, OIL].sum() * 1000
                if len(LATEWATER)>3:
                    OUTPUT.at[UWI,'OWC_Final'] =  pdf.loc[LATEWATER, OIL].sum() / (pdf.loc[LATEWATER, OIL].sum()+pdf.loc[LATEWATER, WTR].sum())
                
                OUTPUT.at[UWI,'Production_Formation'] = '_'.join(pdf[FM].astype(str).str.replace(r'\d','',regex = True).replace('',np.nan).dropna().unique())

            if len(pdf[DATE].dropna())>10:
                MONTH1 = pdf.loc[(pdf[DAYSON]>14) & (pdf[OIL]>0),DATE].min()
                OUTPUT.at[UWI,'Month1'] = MONTH1

                if not isinstance(MONTH1,float):
                    pdf['EM_PRODMONTH'] = (pd.to_datetime(pdf[DATE]).dt.year - MONTH1.year)*12+(pd.to_datetime(pdf[DATE]).dt.month - MONTH1.month)+1

                    for i in MonthArray:
                        if max(pdf['EM_PRODMONTH']) >= i:
                            i_dwn = i-1
                            i_up = i+1
                            OUTPUT.at[UWI,'CumOil_Mo'+str(i)] = pdf.loc[(pdf['EM_PRODMONTH']<=i),OIL].sum()
                            OUTPUT.at[UWI,'CumGas_Mo'+str(i)] = pdf.loc[(pdf['EM_PRODMONTH']<=i),GAS].sum()
                            OUTPUT.at[UWI,'CumWtr_Mo'+str(i)] = pdf.loc[(pdf['EM_PRODMONTH']<=i),WTR].sum()
                                    
                            if pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum() > 0:
                                OUTPUT.at[UWI,'GOR_MO'+str(i_dwn)+'_'+str(i_up)]  = pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),GAS].sum()*1000 / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum()
                            if (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),WTR].sum()) >=1:   
                                OUTPUT.at[UWI,'OWC_MO'+str(i)] = pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() / (pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),OIL].sum() + pdf.loc[(pdf['EM_PRODMONTH']>=0) & (pdf['EM_PRODMONTH']<=i),WTR].sum())
                            if pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),WTR].sum() > 0: 
                                OUTPUT.at[UWI,'OWR_MO'+str(i_dwn)+'_'+str(i_up)]  = pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),OIL].sum() / pdf.loc[(pdf['EM_PRODMONTH']>=i_dwn) & (pdf['EM_PRODMONTH']<=i_up),WTR].sum()

    # CALCULATE SLOPES
    # Should add check for additional completions
    mm_nOIL = ppdf.index[(ppdf['NORM_OIL']>0.1)*(ppdf['NORM_OIL']<0.9)]
    mm_nGAS = ppdf.index[(ppdf['NORM_GAS']>0.1)*(ppdf['NORM_GAS']<0.9)]
    mm_nWTR = ppdf.index[(ppdf['NORM_WTR']>0.1)*(ppdf['NORM_WTR']<0.9)]
    
    # CALCULATE MODEL_PARAMETERS
    mm_o95 = ppdf.index[ppdf['NORM_OIL']<0.95]
    mm_g95 = ppdf.index[ppdf['NORM_GAS']<0.95]
    mm_w95 = ppdf.index[ppdf['NORM_WTR']<0.95]
    mm_otmb200 = ppdf.index[ppdf['TMB_OIL']<200]
    mm_gtmb200 = ppdf.index[ppdf['TMB_GAS']<200]
    mm_wtmb200 = ppdf.index[ppdf['TMB_WTR']<200]
    PAIRS = [('TMB_OIL','NORM_OIL', True, False, mm_o95.intersection(mm_otmb200), stretch_exponential),
             ('TMB_GAS','NORM_GAS', True, False, mm_g95.intersection(mm_gtmb200), stretch_exponential),
             ('TMB_WTR','NORM_WTR', True, False, mm_w95.intersection(mm_otmb200), stretch_exponential),
             ('TMB_GAS','CUM_GOR', True, False, mm_g95, sigmoid),
             ('TMB_WTR','OWC', True, False, mm_w95, sigmoid)]
    MODELS = pd.DataFrame()
    for (Xkey, Ykey, logx_bool, logy_bool, mm, func) in PAIRS:
            try:
                MODEL = ppdf.loc[mm,[UWIKEY ,Xkey,Ykey]].replace((np.inf,-np.inf,None),np.nan).dropna(how='any', axis = 0).groupby([UWIKEY ]).apply(lambda x: curve_fitter(x[Xkey],x[Ykey], funct = func, split = None, plot = False, logx = logx_bool, logy = logy_bool))
                params = int(MODEL.dropna().apply(len).mode())
                MODEL.dropna(inplace = True)
                NAME = '_'.join([Xkey,Ykey])
                param_names = [f'{NAME}_{func.__name__}_{x}' for x in np.arange(0,params)]
                MODEL2 = pd.DataFrame(MODEL.tolist(), columns = param_names, index = MODEL.index)

                #MODELS[NAME] = MODEL
                MODELS = pd.merge(MODELS,MODEL2,how = 'outer',left_index = True,right_index=True)
            except:
                pass
    MODELS.reset_index(drop=False, inplace = True)
    if UWIKEY != 'UWI10':
        MODELS['UWI10'] = MODELS[UWIKEY].apply(lambda x:WELLAPI(x).API2INT(10))
        MODELS.drop(UWIKEY, axis= 1, inplace = True)
    
    #OUTPUT = pd.concat([OUTPUT,MODELS], axis = 0, join = 'outer') # left_index = True, right_index = True, how= 'outer')
    if MODELS.shape[0]>0:
        OUTPUT = pd.merge(OUTPUT,MODELS, how = 'left', on = 'UWI10')
    #OUTPUT.at[UWI,'Production_Formation'] = '_'.join(pdf[FM].unique())
           
    if ADD_RATIOS:
        return OUTPUT, ppdf
    else:
        return OUTPUT


def CO_Get_Surveys(UWIx,URL_BASE:str = 'https://ecmc.state.co.us/cogisdb/Resources/Docs?id=XNUMBERX',DL_BASE:str = 'https://ecmc.state.co.us/weblink/DownloadDocumentPDF.aspx?DocumentId=XLINKX', FOLDER = None, REPLACE:bool = False):
           
    #URL_BASE = 'http://cogcc.state.co.us/weblink/results.aspx?id=XNUMBERX'
    #DL_BASE = 'http://cogcc.state.co.us/weblink/XLINKX'
    #pathname = path.dirname(argv[0])       
    #adir = path.abspath(pathname)
    adir = getcwd()
    if FOLDER == None:      
        dir_add = path.join(adir,"SURVEYS")
        if path.isdir(dir_add) == False:    
           mkdir(dir_add)
    else:
        if path.exists(FOLDER):
           dir_add= FOLDER
        elif path.exists(path.join(adir,FOLDER)):
           dir_add = path.join(adir,FOLDER)
        else:
           dir_add = path.join(adir,FOLDER)
           mkdir(dir_add)
           
    if isinstance(UWIx,(str,int,float)):
        UWIx=[UWIx]
    if isinstance(UWIx,(np.ndarray,pd.Series,pd.DataFrame)):
        UWIx=pd.DataFrame(UWIx).iloc[:,0].tolist()

    UWIx = [WELLAPI(x).STRING(10) for x in UWIx]
           
    with get_driver() as browser:
        for UWI in UWIx:
            #print(UWI)
            UWI = WELLAPI(UWI).STRING(10)
            #warnings.simplefilter("ignore")
            SUCCESS=TRYCOUNT=PAGEERROR=ERROR=0
            while ERROR == 0:
                while (ERROR==0) & (TRYCOUNT<6):
                    TRYCOUNT+=1
                    #print(TRYCOUNT)
                    if TRYCOUNT>1:
                        sleep(10)
                    #Screen for Colorado wells
                    surveyrows=pd.DataFrame()
                    if UWI[0:2] != '05':
                        ERROR = 1
                    #Reduce well to county and well numbers
                    COWELL=UWI[2:10]
                    docurl=re.sub('XNUMBERX',COWELL,URL_BASE)
                    #page = requests.get(docurl)
                    #if str(page.status_code)[0] == '2':
                    #ADD# APPEND ERROR CODE TO LOG FILE
                    #service = service.Service('\\\Server5\\Users\\KRucker\\chromedriver.exe')
                    #service.start()
                    #capabilities = {'chrome.binary': "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"}
                    #options = Options();
                    #options.setBinary("C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe")
                    
                    #options.setBinary("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe")
                    #option.add_argument(' — incognito')
                    #browser = webdriver.Chrome('\\\Server5\\Users\\KRucker\\chromedriver.exe')
                    
                    try:
                        browser.get(docurl)
                    except Exception as ex:
                        print(f'Error connecting to {docurl}.')
                        print(ex)
                        ERROR=1
                        continue
                    
                    browser.find_element(By.LINK_TEXT, "Document Name").click()

                    soup = BS(browser.page_source, 'lxml')
                    
                    try:
                        parsed_table = soup.find_all('table')[0]
                    except Exception as ex:
                        print(f'Error parsing {docurl}')
                        print(ex)
                        continue
                
                    pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)
                    pdf = [p for p in pdf if 'Download' in p.keys()][0]
                    links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
                    pdf['LINK']=None
                    pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links

                    #surveyrows=pdf.loc[(pdf.iloc[:,3].astype(str).str.contains('DIRECTIONAL DATA' or 'DEVIATION SURVEY DATA' or 'DIRECTIONAL SURVEY' or 'GYRO SURVEY', case = False)==True)]
                    m = pdf.iloc[:,3].str.contains('(?=.*DIRECTIONAL|.*DEVIAT|.*GYRO)(?=.*DATA|.*SURVEY)',flags = re.I,regex= True).fillna(False)
                    surveyrows = pdf.loc[m,:]
           
                    # If another page, scan it too
                    # select next largest number
                    tables=len(soup.find_all('table'))
                    parsed_table = soup.find_all('table')[tables-1]
                    data = [[td.a['href'] if td.find('a') else
                             '\n'.join(td.stripped_strings)
                            for td in row.find_all('td')]
                            for row in parsed_table.find_all('tr')]
                    pages=len(data[0])

                    if pages>0:
                        for p in range(1,pages):
                            browser.find_element(by = 'link text',value = str(1+p)).click()
                            browser.page_source
                            soup = BS(browser.page_source, 'lxml')
                            #check COGCC site didn't glitch
                            tables=len(soup.find_all('table'))
                            parsed_table = soup.find_all('table')[tables-1]
                            data = [[td.a['href'] if td.find('a') else
                                 '\n'.join(td.stripped_strings)
                                for td in row.find_all('td')]
                                for row in parsed_table.find_all('tr')]
                            newpages=len(data[0])
                            if newpages>pages:
                                PAGEERROR=1
                                break
                            parsed_table = soup.find_all('table')
                            for i,p in enumerate(parsed_table):
                                try:
                                    pdf = pd.read_html(str(p),encoding='utf-8', header=0)
                                    pdf = [p for p in pdf if 'Download' in p.keys()][0]
                                    break       
                                except:
                                    pass
                            parsed_table = soup.find_all('table')[i]
                            links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]
                            pdf['LINK']=None
                            pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links
                            #dirdata=[s for s in data if any(xs in s for xs in ['DIRECTIONAL DATA','DEVIATION SURVEY DATA'])]
                            #surveyrows.append(dirdata)
                            #surveyrows.append(pdf.loc[pdf.iloc[:,3].astype(str).str.contains('DIRECTIONAL DATA' or 'DEVIATION SURVEY DATA' or 'DIRECTIONAL SURVEY' or 'GYRO SURVEY', case = False)==True])
                            m = pdf.iloc[:,3].str.contains('(?=.*DIRECTIONAL|.*DEVIAT|.*GYRO)(?=.*DATA|.*SURVEY)',flags = re.I,regex= True).fillna(False)
                            surveyrows = pd.concat([surveyrows,pdf.loc[m,:]], ignore_index = True, axis = 0)    
                    elif (pages == 0) and (sum([len(i) for i in data]) > 10):

                        parsed_table = soup.find_all('table')[tables-1]
                        pdf = pd.read_html(str(parsed_table),encoding='utf-8', header=0)
                        # get DL table:
                           
                        pdf = [p for p in pdf if 'Download' in p.keys()][0]
                        links = [np.where(tag.has_attr('href'),tag.get('href'),"no link") for tag in parsed_table.find_all('a',string='Download')]   
                        pdf['LINK']=None
                        pdf.loc[pdf.Download.str.lower()=='download',"LINK"]=links
                        #dirdata=[s for s in data if any(xs in s for xs in ['DIRECTIONAL DATA','DEVIATION SURVEY DATA'])]
                        #surveyrows.append(dirdata)
                        #surveyrows.append(pdf.loc[pdf.iloc[:,3].astype(str).str.contains('DIRECTIONAL DATA' or 'DEVIATION SURVEY DATA' or 'DIRECTIONAL SURVEY' or 'GYRO SURVEY', case = False)==True])
                        m = pdf.iloc[:,3].str.contains('(?=.*DIRECTIONAL|.*DEVIAT|.*GYRO)(?=.*DATA|.*SURVEY)',flags = re.I,regex= True).fillna(False)
                        surveyrows = pd.concat([surveyrows,pdf.loc[m,:]], ignore_index = True, axis = 0)
                    else:
                        print(f'No Tables for {UWI}')
                        PAGEERROR=ERROR=1
                        break          
                    surveyrows=pd.DataFrame(surveyrows)
                    m = surveyrows.iloc[:,:-1].drop_duplicates().index
                    surveyrows = surveyrows.loc[m]
                    if len(surveyrows)==0:
                        ERROR=1
                        break
                    surveyrows.loc[:,'DateString']=None
                    surveyrows.loc[:,'DateString']=surveyrows['Date'].astype('datetime64[ns]').dt.strftime('%Y_%m_%d')
                    LINKCOL=surveyrows.columns.get_loc('LINK')
                    for i in range(0,surveyrows.shape[0]):
                        #dl_url= re.sub('XLINKX', str(surveyrows.loc[surveyrows['Date'].astype('datetime64').idxmax(),'LINK']),DL_BASE)
                        #DocDate=str(surveyrows.loc[surveyrows['Date'].astype('datetime64').idxmax(),'DateString'])
                        dl_url= re.sub('XLINKX', str(surveyrows.iloc[i,LINKCOL]),DL_BASE)
                        DocDate=str(surveyrows.iloc[i,surveyrows.columns.get_loc('DateString')])
                        DocID = re.findall('DocumentId=(.*)',str(surveyrows.iloc[i,LINKCOL]))[-1]
                        #df=pd.DataFrame(surveyrows[0]).transpose()
                        #daterow=df[0].str.contains("DocDate")
                        ##df.loc[df[0].str.contains("DocDate"),:]
                        #df.loc[df[0].str.contains("DocDate"),:].replace({r'.*([0-9]{2}/[0-9]{2}/[0-9]{2,4}).*':r'\1'},regex=True).astype('datetime64')          
                        #with requests.get(dl_url) as r:
                        #    soup = BS(r.content, 'lxml')
                        #    parsed_table = soup.find_all('table')[0]
                        #    data = [[td.a['href'] if td.find('a') else
                        #             ''.join(td.stripped_strings)
                        #            for td in row.find_all('td')]
                        #            for row in parsed_table.find_all('tr')]
                        #    dirdata=[s for s in data if any(xs in s for xs in ['DIRECTIONAL DATA','DEVIATION SURVEY DATA'])]
                        #    surveyrows.append(dirdata)
                        #    df = pd.DataFrame(data[1:], columns=data[0])
                        #    # If another page, scan it too
                        #    parsed_table = soup.find_all('table')[0]
                        #Select most recent survey data and download   
                        #XX=df.loc[df[0].str.contains("DocDate"),:].replace({r'.*([0-9]{2}/[0-9]{2}/[0-9]{2,4}).*':r'\1'},regex=True).astype('datetime64').transpose().idxmax()
                        #dl_url=re.sub('XLINKX',df.loc[df[0].str.contains("Download"),int(XX)].to_string(index=False),DL_BASE)
                        ecmc_file_url = 'http'+dl_url.split('http')[-1]
                        r=requests.get(ecmc_file_url, allow_redirects=True)
                        filetype=path.splitext(re.sub(r'.*filename=\"(.*)\"',r'\1',r.headers['content-disposition']))[1]
                        filename=path.join(dir_add,'SURVEYDATA_'+DocDate+'_DOCID'+str(DocID)+'_UWI'+str(UWI)+filetype)
                        if REPLACE:                            
                            if path.exists(filename):
                                remove(filename)
                        if not path.exists(filename):
                            urllib.request.urlretrieve(ecmc_file_url, filename)
                        # filename=dir_add+'\\SURVEYDATA_'+DocDate+'_'+str(UWI)+'_1'+filetype
                        #urllib.request.urlretrieve(ecmc_file_url, filename)
                        #urllib.request.urlretrieve(dl_url, filename)
                    SUCCESS=1
                    if PAGEERROR==1:
                         #TRYCOUNT+=1
                         PAGEERROR=0
                    if SUCCESS==1:
                        ERROR = 1
    try: browser.quit()
    except Exception:
        None

def SUMMARIZE_COGCC_SQL(SAVE = True, SAVEDB= 'FIELD_DATA.db',TABLE_NAME = 'CO_SQL_SUMMARY'):
    # SQLite COMMMANDS
    # well data
    Q1 = """
        SELECT
                P.StateProducingZoneKey
                ,P.ProductionDate
                ,P.StateProducingUnitKey
                ,P.DaysProducing
                ,P.Oil
                ,P.Gas
                ,P.Water
                ,SUM(CASE WHEN P.DaysProducing + 0 = 0 THEN 28
                           ELSE P.DaysProducing 
                           END) 
                    OVER(PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as DaysOn
                ,CAST(SUM(CASE WHEN P.DaysProducing + 0 = 0 THEN 28
                           ELSE P.DaysProducing
                           END) 
                  OVER(PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC)/30.4 AS int) AS MonthsOn
                ,SUM(P.Oil+0) OVER 
                        (PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as CumOil
                ,SUM(P.Gas+0) OVER 
                        (PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as CumGas        
                ,SUM(P.Water+0) OVER 
                        (PARTITION BY P.StateProducingUnitKey
                        ORDER BY P.ProductionDate ASC) as CumWater
                ,P.GAS / P.OIL *1000 as GOR
                ,P.OIL / IIF(P.DaysProducing is null,28,P.DaysProducing) AS OilRate_BOPD
                ,P.Gas / IIF(P.DaysProducing is null = 0,28,P.DaysProducing) AS GasRate_MCFPD
                ,P.Water / IIF(P.DaysProducing is null = 0,28,P.DaysProducing) AS WaterRate_BWPD
                ,((P.GAS / P.OIL - LAG(P.GAS,2) OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                        / LAG(P.OIL,2)  OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                                ) +
                        (P.GAS / P.OIL - LAG(P.GAS,1) OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                        / LAG(P.OIL,1)  OVER (
                                Partition By P.StateProducingUnitKey 
                                Order By P.ProductionDate)
                                ))*1000
                        AS dGOR
                ,PM.StartDate
                ,PM.PeakOil
                ,PM.PeakGas
                ,PM.PeakWater
                ,P.Oil/P.DaysProducing / PM.PeakOil AS NormOilRate
                ,P.Gas/P.DaysProducing / PM.PeakGas AS NormGasRate
                ,P.Water/P.DaysProducing / PM.PeakWater AS NormWater
                ,P.Gas /P.DaysProducing /  PM.PeakGas / (P.Oil / P.DaysProducing / PM.PeakOil) AS NormGOR
                ,(0+STRFTIME('%Y',PM.StartDate)) as MINYEAR
                ,W.*
        FROM Production P
        LEFT JOIN Well AS W ON W.StateWellKey = P.StateProducingUnitKey
        LEFT JOIN (
                select StateProducingUnitKey
                        , MIN(ProductionDate) as StartDate
                        , MAX(Oil/DaysProducing) AS PeakOil
                        , MAX(Gas/DaysProducing) AS PeakGas
                        ,MAX(Water/DaysProducing) AS PeakWater 
                        from PRODUCTION 
                        WHERE DaysProducing < 600
                        GROUP BY StateProducingUnitKey
                ) PM
                ON PM.StateProducingUnitKey = P.StateProducingUnitKey
        WHERE (P.OIL + P.WATER + P.GAS) >0 
        --      AND (STRFTIME('%Y',PM.StartDate)+0) > 2015
        --      AND substr(P.StateProducingUnitKey,1,7)+0 >=  512334
        --  AND (0+STRFTIME('%Y',MIN(ProductionDate) OVER (PARTITION BY StateProducingUnitKey)))>=2015
        --      AND Trim(P.StateProducingZoneKey) = "NBRR"
        -- GROUP BY StateProducingUnitKey
        ORDER BY P.StateProducingUnitKey,P.ProductionDate DESC

    """

    Q1 = """
    SELECT
        StateProducingUnitKey
        ,MIN(ProductionDate) as StartDate
        ,MAX(Oil/DaysProducing) AS PeakOil
        ,MAX(Gas/DaysProducing) AS PeakGas
        ,MAX(Water/DaysProducing) AS PeakWater
        FROM PRODUCTION 
        WHERE DaysProducing < 800
        GROUP BY StateProducingUnitKey
    """

    # Completions Groups
    Q2_Drop = "DROP TABLE If EXISTS temp_table1;"
    Q2 = """

    CREATE TABLE IF NOT EXISTS temp_table1 AS
        SELECT DISTINCT
            StateProducingUnitKey
            ,CAST(StateProducingUnitKey AS INTEGER) AS INT_SPUKEY
            ,ProductionDate
            ,SUM(Oil + Gas + Water) OVER (PARTITION BY  StateProducingUnitKey
                                                ORDER BY ProductionDate
                                                ROWS BETWEEN UNBOUNDED PRECEDING    
                                                    AND 1 PRECEDING) 
                AS Cum_OilGasWtr
        FROM Production
        WHERE (Oil + Gas + Water)>=0
        GROUP BY StateProducingUnitKey,ProductionDate;

    SELECT 
        S.StateWellKey
        ,S.StateStimulationKey
        ,S.TreatmentDate
        ,TR2.PastTreatmentDate
        ,TR2.PastStateStimulationKey
        ,P1.Cum_OilGasWtr as CumTreatOne
        ,P2.Cum_OilGasWtr as CumTreatTwo
        ,P1.Cum_OilGasWtr - P2.Cum_OilGasWtr as Cum_Difference
        , DATE(TR.TreatmentDate,'start of month') AS DATE1
        , DATE(TR2.PastTreatmentDate,'start of month') AS DATE2
        ,TR.TreatmentRank as TreatmentRank
    FROM Stimulation S
    LEFT JOIN (
        SELECT 
            StateStimulationKey
            ,StateWellKey
            ,TreatmentDate
            ,Rank() OVER (
                PARTITION BY StateWellKey
                ORDER BY TreatmentDate ASC) TreatmentRank
            ,COUNT(TreatmentDate) Over (PARTITION BY StateWellKey) as TreatmentCount
        FROM Stimulation 
        ) TR
            ON TR.StateStimulationKey = S.StateStimulationKey
    LEFT JOIN (
        SELECT 
            StateStimulationKey AS PastStateStimulationKey
            ,StateWellKey
            ,TreatmentDate As PastTreatmentDate
            ,Rank() OVER (
                PARTITION BY StateWellKey
                ORDER BY TreatmentDate ASC) PastTreatmentRank
        FROM Stimulation ) TR2
    ON  (TR2.PastTreatmentRank + 1)= TR.TreatmentRank AND TR2.StateWellKey=TR.StateWellKey          
    LEFT JOIN temp_table1 P1
        ON  P1.INT_SPUKEY = CAST(S.StateWellKey AS INTEGER)
            AND DATE(P1.ProductionDate,'start of month') = DATE(TR.TreatmentDate,'start of month')
    LEFT JOIN temp_table1 P2
        ON  P2.INT_SPUKEY = CAST(S.StateWellKey AS INTEGER)
            AND DATE(P2.ProductionDate,'start of month') = DATE(TR2.PastTreatmentDate,'start of month')
    ORDER BY Cum_Difference DESC
    """

    #Completion Volumes
    Q3 = """

    SELECT 
            S.*
            ,SF.Type
            ,SF.TotalAmount
            ,SF.QC_CountUnits
        FROM Stimulation S
        LEFT JOIN(
            SELECT 
                StateStimulationKey
                ,FluidType as Type
                ,SUM(Volume) as TotalAmount
                ,COUNT(DISTINCT FluidUnits) as QC_CountUnits
            FROM StimulationFluid
            GROUP BY StateStimulationKey,FluidType
            ) SF
            ON SF.StateStimulationKey = S.StateStimulationKey
    UNION   
    SELECT 
            S.*
            ,SP.Type
            ,SP.TotalAmount
            ,SP.QC_CountUnits
        FROM Stimulation S      
            LEFT JOIN(
            SELECT
                StateStimulationKey
                ,'PROPPANT' as Type
                ,SUM(ProppantAmount) as TotalAmount
                ,COUNT(DISTINCT ProppantUnits) as QC_CountUnits
            FROM StimulationProppant
            GROUP BY StateStimulationKey
            ) SP
            ON SP.StateStimulationKey = S.StateStimulationKey   
    """

    # 3,6,9,12,15,18,21,24 month cum
    Q4="""
    SELECT 
        PP.StateProducingUnitKey
        ,Min(CASE
        WHEN PP.Oil + PP.Gas + PP.Water >=0
        THEN PP.ProductionDate  
        Else NULL
        END)  as FirstProduction 
        ,Min(CASE 
            WHEN PP.OilRank=1
            THEN PP.CumOil
            Else NULL
            END
            ) AS PeakOil_CumOil
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.CumGas
            Else NULL
            END
            )  AS PeakOil_CumGas
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.CumWater
            Else NULL
            END
            )  AS PeakOil_CumWater 
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.ProductionDate
            Else NULL
            END
            )  AS PeakOil_Date
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.DaysOn
            Else NULL
            END
            )  AS PeakOil_DaysOn
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.OilRate_BBLPD
            Else NULL
            END
            )  AS PeakOil_OilRate
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.GasRate_MCFPD
            Else NULL
            END
            )  AS PeakOil_GasRate
        ,Min(CASE PP.OilRank
            WHEN 1
            THEN PP.WaterRate_BBLPD
            Else NULL
            END
            )  AS PeakOil_WaterRate
        ,Min(CASE 
            WHEN PP.WaterRank=1
            THEN PP.CumOil
            Else NULL
            END
            ) AS PeakWater_CumOil
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.CumGas
            Else NULL
            END
            )  AS PeakWater_CumGas
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.CumWater
            Else NULL
            END
            )  AS PeakWater_CumWater 
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.ProductionDate
            Else NULL
            END
            )  AS PeakWater_Date
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.DaysOn
            Else NULL
            END
            )  AS PeakWater_DaysOn
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.OilRate_BBLPD
            Else NULL
            END
            )  AS PeakWater_OilRate
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.GasRate_MCFPD
            Else NULL
            END
            )  AS PeakWater_GasRate
        ,Min(CASE PP.WaterRank
            WHEN 1
            THEN PP.WaterRate_BBLPD
            Else NULL
            END
            )  AS PeakWater_WaterRate
        ,Min(CASE 
            WHEN PP.GasRank=1
            THEN PP.CumOil
            Else NULL
            END
            ) AS PeakGas_CumOil
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.CumGas
            Else NULL
            END
            )  AS PeakGas_CumGas
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.CumWater
            Else NULL
            END
            )  AS PeakGas_CumWater
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.ProductionDate
            Else NULL
            END
            )  AS PeakGas_Date
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.DaysOn
            Else NULL
            END
            )  AS PeakGas_DaysOn
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.OilRate_BBLPD
            Else NULL
            END
            )  AS PeakGas_OilRate
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.GasRate_MCFPD
            Else NULL
            END
            )  AS PeakGas_GasRate
        ,Min(CASE PP.GasRank
            WHEN 1
            THEN PP.WaterRate_BBLPD
            Else NULL
            END
            )  AS PeakGas_WaterRate    
         ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*3)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '3MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*6)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '6MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*9)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '9MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*12)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '12MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*15)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '15MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*18)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '18MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*21)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '21MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*24)
            THEN PP.CumOil
            Else NULL
            END
            ) AS '24MonthOil'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*3)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '3MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*6)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '6MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*9)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '9MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*12)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '12MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*15)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '15MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*18)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '18MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*21)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '21MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*24)
            THEN PP.CumGas
            Else NULL
            END
            ) AS '24MonthGas'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*3)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '3MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*6)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '6MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*9)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '9MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*12)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '12MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*15)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '15MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*18)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '18MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*21)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '21MonthWater'
        ,MIN(CASE 
            WHEN (PP.DaysOn+1) >=(30.4*24)
            THEN PP.CumWater
            Else NULL
            END
            ) AS '24MonthWater'
    FROM (
    SELECT
        P.*
        ,RANK() OVER (PARTITION BY P.StateProducingUnitKey
                    ORDER BY P.OilRate_BBLPD DESC) as OilRank
        ,RANK() OVER (PARTITION BY P.StateProducingUnitKey
                    ORDER BY P.GasRate_MCFPD DESC) as GasRank
        ,RANK() OVER (PARTITION BY P.StateProducingUnitKey
                    ORDER BY P.WaterRate_BBLPD DESC) as WaterRank
    FROM(
        SELECT 
                    StateProducingUnitKey   
                    ,Max(ProductionDate) as ProductionDate
                    ,Max(DaysProducing) as DaysProducing
                    ,Sum(Oil) as Oil
                    ,Sum(Gas) as Gas
                    ,Sum(Water) as Water
                    ,Sum(CO2) as CO2
                    ,Sum(Sulfur) as Sulfur
                    ,Sum(Oil) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) CumOil
                    ,Sum(Gas) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) CumGas
                    ,Sum(Water) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) CumWater      
                    ,Sum(MAX(DaysProducing)) OVER (Partition By StateProducingUnitKey 
                                    Order By ProductionDate
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING) DaysOn
                    ,Sum(Oil) / MAX(DaysProducing) as OilRate_BBLPD
                    ,Sum(Gas) / MAX(DaysProducing) as GasRate_MCFPD
                    ,Sum(Water) / MAX(DaysProducing) as WaterRate_BBLPD
                    FROM PRODUCTION
    --                  WHERE CAST(StateProducingUnitKey as INTEGER) >= 5123449960000 AND CAST(StateProducingUnitKey as INTEGER) <= 5123450000000
                    GROUP BY StateProducingUnitKey, ProductionDate
                    ORDER BY  StateProducingUnitKey,ProductionDate ASC
                    ) P
    WHERE P.DaysOn <= 700
    ORDER BY  StateProducingUnitKey,ProductionDate ASC) PP
    -- WHERE CAST(PP.StateProducingUnitKey as INTEGER) >= 5123449960000 AND CAST(PP.StateProducingUnitKey as INTEGER) <= 5123450000000
    GROUP BY PP.StateProducingUnitKey
    """
    #quit()

    sqldb = "CO_3_2.1.sqlite"
    conn = sqlite3.connect(sqldb)
    c = conn.cursor()
    # c.execute(Query1)

    Well_df=pd.read_sql_query(Q1,conn)

    # Determine how to aggregate treatments
    if 1==1:
        c = conn.cursor()
        c.execute(Q2_Drop)
        c.execute(Q2.split(';')[0])
       # c.execute(Q2.split(';')[1)
       # df = pd.read_sql_query(Q2.split(';')[2],conn)
        df = pd.read_sql_query(Q2.split(';')[1],conn)
        df.loc[df.PastTreatmentDate!=None]
        df2 = pd.merge(df,pd.DataFrame(df).pivot_table(index=['StateWellKey'],aggfunc='size').rename('Count'),how='left',left_on='StateWellKey',right_on='StateWellKey')
        df2['TDelta']=(pd.to_datetime(df2.DATE1)-pd.to_datetime(df2.DATE2)).astype('timedelta64[M]')
        x=df2.loc[(df2.Cum_Difference<50) | ((df2.TDelta <6) & (df2.Cum_Difference<1000)) | (df2.TDelta < 2)].sort_values(by='Cum_Difference')
        df2['TreatKey2']=0
        df2['TreatKey2']=((df2.Cum_Difference<50) | ((df2.TDelta <6) & (df2.Cum_Difference<1000)) | (df2.TDelta < 2))*1+df2.TreatKey2
        # For Stimulation order A-B-C
        # Dictionary receives stimulation B and returns A
        StimDictPrev = df2.loc[df2.TreatKey2>0].set_index('StateStimulationKey').PastStateStimulationKey.to_dict()
        # Dictionary receives stimulation B and returns C
        StimDict = df2.loc[df2.TreatKey2>0].set_index('PastStateStimulationKey').StateStimulationKey.to_dict()                          
        # StimDict to map and group
        # df2.loc[df2.TreatKey2>0].PastStateStimulationKey.map(StimDict).dropna()
        del df,df2,x                              

    # Get Stimulation Quantities
    if 1==1:
        STIM_df=pd.read_sql_query(Q3,conn)
        t_df = STIM_df.pivot(index = ['StateStimulationKey'], columns = ['Type'],values = ['TotalAmount','QC_CountUnits']).dropna(axis=1,how='all').dropna(axis=0,how='all')
        t_df.columns = ['_'.join(col).strip() for col in t_df.columns.values]

        STIM_df[['StateStimulationKey','TreatmentDate','StateWellKey']].drop_duplicates()

        STIM_df = pd.merge(t_df
            ,STIM_df[['StateStimulationKey','TreatmentDate','StateWellKey']].drop_duplicates()
            ,how = 'left'
            ,on = ['StateStimulationKey','StateStimulationKey'])
        del t_df
        i = 1
        STIM_df=STIM_df.drop(list(filter(re.compile('StateStimulationKey[0-9]').match, STIM_df.keys().to_list())),axis=1)    
        STIM_df['StateStimulationKey0']=STIM_df['StateStimulationKey']
        while STIM_df['StateStimulationKey'+str(i-1)].map(StimDict).dropna().shape[0]>0:
            STIM_df['StateStimulationKey'+str(i)] = STIM_df['StateStimulationKey'+str(i-1)]
            #STIM_df['StateStimulationKey'+str(i-1)].map(StimDict)
            m = STIM_df['StateStimulationKey'+str(i-1)].map(StimDict).dropna().index
            STIM_df.loc[m, 'StateStimulationKey'+str(i)] = STIM_df.loc[m,'StateStimulationKey'+str(i-1)].map(StimDict)
            i+=1
        # Stimulation Dates Dictionary
        #StimDates=STIM_df[['StateStimulationKey','TreatmentDate']].drop_duplicates().set_index('StateStimulationKey').astype('datetime64').TreatmentDate.to_dict()
        StimDates = STIM_df[['StateStimulationKey','TreatmentDate']].drop_duplicates().set_index('StateStimulationKey')       
        StimDates = StimDates.apply(lambda x:np.datetime64(x[0]), axis = 1)
        StimDates = StimDates.to_dict()       
        # Count of stimulations per well


    ##################################################################################################      UPDATE ME
        # UPDATE THIS TO TRACK INTERVAL OF COMPLETION ACTIVITY
        # AND ALSO FOLLOW THESE INTERVALS IN PRODUCTION
    ##################################################################################################
    ## Not sure, but I think this whole section has been superceded by lower sections
    ##if 1==1:
    ##    # Sum stimulation volumes where multiple stimulations in a short time
    ##    sumdf = STIM_df[(list(filter(re.compile('Total.*').match, STIM_df.keys().to_list())))+['StateStimulationKey'+str(i-1)]].groupby('StateStimulationKey'+str(i-1)).sum()    
    ##    # Max for QC Counts
    ##    maxdf = STIM_df[(list(filter(re.compile('QC_CountUnit.*').match, STIM_df.keys().to_list())))+['StateStimulationKey'+str(i-1)]].groupby('StateStimulationKey'+str(i-1)).max()
    ##    #Stimulation Summary Table
    ##    SS_df=sumdf.join(maxdf, on='StateStimulationKey'+str(i-1))
    ##    # Add Well API
    ##    SS_df.join(STIM_df[['StateStimulationKey'+str(i-1),'StateWellKey']].set_index('StateStimulationKey'+str(i-1)).drop_duplicates())
    ##    del x,maxdf,sumdf

    if 1==1:
        STIM_df['LastStimDate']=STIM_df['StateStimulationKey'+str(i-1)].map(StimDates)
        STIM_df[['StateWellKey','StateStimulationKey'+str(i-1),'LastStimDate']].drop_duplicates()
        ranks = STIM_df[['StateWellKey','StateStimulationKey'+str(i-1),'LastStimDate']].drop_duplicates() \
          .groupby(['StateWellKey'])['LastStimDate'] \
          .rank(ascending = True, method = 'first')-1
        ranks = ranks.fillna(0).astype('int64')

        RankDict = pd.Series(ranks.values.astype('int64')
                             ,index=STIM_df[['StateWellKey'
                             ,'StateStimulationKey'+str(i-1)
                             ,'LastStimDate']].drop_duplicates()['StateStimulationKey'+str(i-1)].values).to_dict()

        #ranks.name = 'StimRank'
        STIM_df['StimRank'] = STIM_df['StateStimulationKey'+str(i-1)].map(RankDict)
        STIM_df['StimRank'] = STIM_df['StimRank'].fillna(0)
        STIM_df['API14x']= (STIM_df.StimRank.astype('int64') + STIM_df.StateWellKey.astype('int64')).astype('str').str.zfill(14)

        tbl = STIM_df[['StateWellKey','StateStimulationKey'+str(i-1),'LastStimDate','StimRank']]
        xbl = tbl.loc[tbl.StimRank == tbl.StimRank.max()].set_index('StateWellKey')

        dict1 = {};dict2 = {};
        for j in reversed(range(0,tbl.StimRank.max())):
            #suffix for rank n-1 values
            sfx = '_prev'
            x = pd.merge(tbl.loc[tbl.StimRank == j+1].set_index('StateWellKey')
                 ,tbl.loc[tbl.StimRank == j].set_index('StateWellKey')
                 ,how = 'inner'
                 ,left_index = True
                 ,right_index = True
                 ,suffixes =('',sfx))

            dict1.update(x[['StateStimulationKey'+str(i-1),'StateStimulationKey'+str(i-1)+sfx]].set_index('StateStimulationKey'+str(i-1))['StateStimulationKey'+str(i-1)+sfx].to_dict())
            dict2.update(x[['StateStimulationKey'+str(i-1)+sfx,'StateStimulationKey'+str(i-1)]].set_index('StateStimulationKey'+str(i-1)+sfx)['StateStimulationKey'+str(i-1)].to_dict())
            # dict2.update(dict1)

    if 1==1:
        STIM_df['StimDate0'] = STIM_df['StateStimulationKey'+str(i-1)].map(dict2).map(StimDates).fillna(datetime.date(1900, 1, 1))
        STIM_df['StimDate1'] = STIM_df['StateStimulationKey'+str(i-1)].map(dict1).map(StimDates).fillna(datetime.datetime.now())
        STIM_df.loc[STIM_df.LastStimDate == STIM_df.StimDate0,'StimDate0']=datetime.date(1900, 1, 1)
        STIM_df.loc[STIM_df.StimDate1 == STIM_df.StimDate1,'StimDate1']=datetime.datetime.now()


    # 3,6,9,12,15,18,21,24 month cum
    if 1==1:
        cumdf = pd.read_sql_query(Q4,conn)
        cumdf = cumdf.set_index('StateProducingUnitKey',drop=False)
        #cumdf['StateProducingUnitKey'] = cumdf.index

        # adds empty columns 
        #cumdf = cumdf.reindex(cumdf.columns.tolist() + STIM_df.keys().tolist(), axis=1)
        #cumdf[STIM_df.keys()]=np.nan

        # For speed
        # 1) Assign all wells to 0th conpletion
        # 2) For each well with multiple completions, get assignments
        well_list = cumdf.index.intersection(STIM_df.loc[STIM_df.StimRank>0]['StateWellKey'].unique())

        cumdf = pd.merge(cumdf,STIM_df.drop_duplicates(['StateWellKey']).set_index('StateWellKey'),how='outer',left_index = True, right_index=True)
        STIM_df.StimDate0 = pd.to_datetime(STIM_df.StimDate0)
        cumdf.PeakOil_Date = pd.to_datetime(cumdf.PeakOil_Date)
        for well in well_list:
            x = STIM_df.loc[(STIM_df.StateWellKey == well) \
                & (STIM_df.StimDate0 <= cumdf.loc[well,'PeakOil_Date'])]
            if x.shape[0] ==0:
                continue
            elif x.shape[0] >1:
                x=x.sort_values(by='StimDate0',ascending=False).iloc[0,:]
            else:
                pass
            cumdf.loc[well,STIM_df.keys()]=x.squeeze()

        #StimPivot = STIM_df.pivot_table(values = 'LastStimDate',index = 'StateWellKey',columns = 'StimRank',aggfunc='max')                                                                                                                                        
        #StimPivot.loc[StimPivot.index == '05125121260000'].dropna(axis=1)

        # Get Perforation Detail
        Perf_df = pd.read_sql_query("""SELECT * FROM WellPerf""",conn)
        cumdf=pd.merge(cumdf,Perf_df.drop_duplicates('StateWellKey').set_index('StateWellKey'),how='outer',left_index = True, right_index=True)
        #Perf Interval
        cumdf['PerfInterval'] = cumdf.Bottom-cumdf.Top
        # Frac Per Ft
        cumdf['TotalFluid'] = cumdf['TotalAmount_FRESH WATER']+cumdf['TotalAmount_RECYCLED WATER']
        cumdf['Stimulation_FluidBBLPerFt'] = cumdf['TotalFluid']/cumdf['PerfInterval']
        cumdf['Stimulation_ProppantLBSPerFt'] = cumdf['TotalAmount_PROPPANT']/cumdf['PerfInterval']
        cumdf['Stimulation_ProppantLBSPerBBL'] = cumdf['TotalAmount_PROPPANT']/cumdf['TotalFluid']
        conn.close()

    if 1==1:
        # MERGE WITH WELL HEADERS
        with sqlite3.connect('CO_3_2.1.sqlite') as conn:
            Well_df=pd.read_sql_query("""SELECT * FROM WELL""",conn)
        ULT = pd.merge(cumdf,Well_df.set_index('StateWellKey'), left_index=True, right_index=True,how='outer')
        for k in ULT.keys():
            if 'DATE' in k.upper():
                ULT[k] = pd.to_datetime(ULT['FirstCompDate'])

        #ULT.to_csv('SQL_WELL_SUMMARY.csv')
        ULT.to_json('SQL_WELL_SUMMARY.JSON')
        ULT.to_parquet('SQL_WELL_SUMMARY.PARQUET')


        # dtypes for SQLITE3
        pd_sql_types={'object':'TEXT',
                      'int64':'INTEGER',
                      'float64':'FLOAT',
                      'bool':'TEXT',
                      'datetime64':'TEXT',
                      'datetime64[ns]':'TEXT',
                      'timedelta[ns]':'REAL',
                      'category':'TEXT'
            }

        #dtypes for SQLALCHEMY
        pd_sql_types={'object':sqlalchemy.types.Text,
                      'int64':sqlalchemy.types.BigInteger,
                      'float64':sqlalchemy.types.Float,
                      'bool':sqlalchemy.types.Boolean,
                      'datetime64':sqlalchemy.types.Date,
                      'datetime64[ns]':sqlalchemy.types.Date,
                      'timedelta[ns]':sqlalchemy.types.Float,
                      'category':sqlalchemy.types.Text
            }

        df_typemap = ULT.dtypes.astype('str').map(pd_sql_types).to_dict()

        # ULT replace Nan with None
    ##    ULT2 = ULT.where(pd.notnull(ULT), None)
    ##    with sqlite3.connect('prod_data.db') as conn:
    ##        TABLE_NAME = 'Well_Summary'
    ##        ULT.to_sql(TABLE_NAME,conn,
    ##                      if_exists='replace',
    ##                      index=False,
    ##                      dtype=df_typemap)
        ULT = DF_UNSTRING(ULT)
        SCHEMA = FRAME_TO_SQL_TYPES(ULT)
           
        if SAVE:
            CONN = sqlite3.connect(SAVEDB)
            ULT.to_sql(TABLE_NAME,
                       CONN,
                       if_exists='replace',
                       index=False,
                       dtype=SCHEMA)
    return ULT
#!! not all rows have a state producing key

def PROD_FEATURES(df_in):
    KEYS = {}
    KEYS['OIL'] = GetKey(df_in,r'(?=.*oil)(?=.*(vol|prod))')[0]
    KEYS['GAS'] = GetKey(df_in,r'(?=.*GAS)(?=.*(vol|prod))')[0]

    KEYS['WTR'] = GetKey(df_in,r'(?=.*wa*te*r)(?=.*(vol|prod))')[0]

    KEYS['PROD_DAYS'] = GetKey(df_in,r'(?=.*DAYS)(?=.*PRODUCED)')[0]

    KEYS['DATE'] = GetKey(df_in,r'(?=.*first)(?=.*(month|date))')[0]
    KEYS['UWI'] = GetKey(df_in,r'UWI|API')[0]
    df_in.sort_values(by = [KEYS['UWI'],KEYS['DATE']], inplace = True)
    df_in['OIL_RATE'] = df_in[KEYS['OIL']] / df_in[KEYS['PROD_DAYS']]
    df_in['GAS_RATE'] = df_in[KEYS['GAS']] / df_in[KEYS['PROD_DAYS']]
    df_in['WTR_RATE'] = df_in[KEYS['WTR']] / df_in[KEYS['PROD_DAYS']]
    df_in.sort_values(by = KEYS['DATE'], ascending = True, inplace=True)
    df_in['TOTALDAYS'] = df_in.groupby(['UWI10'])[KEYS['OIL']].cumsum(skipna=True)
    df_in['CUMOIL'] = df_in.groupby(['UWI10'])[KEYS['OIL']].cumsum(skipna=True)
    df_in['CUMGAS'] = df_in.groupby(['UWI10'])[KEYS['GAS']].cumsum(skipna=True)
    df_in['CUMWTR'] = df_in.groupby(['UWI10'])[KEYS['WTR']].cumsum(skipna=True)
    df_in['TMB_OIL'] = df_in['CUMOIL']/df_in[KEYS['OIL']]
    df_in['TMB_GAS'] = df_in['CUMGAS']/df_in[KEYS['GAS']]
    df_in['TMB_WTR'] = df_in['CUMWTR']/df_in[KEYS['WTR']]
    df_in['GOR'] = df_in[KEYS['GAS']] / df_in[KEYS['OIL']] * 1000
    df_in['OWC'] = df_in[KEYS['OIL']] / (df_in[KEYS['OIL']]+df_in[KEYS['WTR']])
    df_in['WOR'] = df_in[KEYS['WTR']] / df_in[KEYS['OIL']]
    df_in['WOC'] = df_in[KEYS['OIL']] / (df_in[KEYS['OIL']]+df_in[KEYS['WTR']])                                        
    df_in['CUMGOR'] = df_in['CUMGAS'] / df_in['CUMOIL'] * 1000
    df_in['CUMOWC'] = df_in['CUMOIL'] / (df_in['CUMOIL']+df_in['CUMWTR'])
    df_in['CUMWOC'] = df_in['CUMWTR'] / (df_in['CUMOIL']+df_in['CUMWTR'])
    df_in['CUMWOR'] = df_in['CUMWTR'] / df_in['CUMOIL']
    return df_in

def ABS_LOC(DB_NAME = 'FIELD_DATA.db'):           
    # CREATE ABSOLUTE LOCATION TABLE if True:
    WELL_LOC = read_shapefile(shp.Reader('Wells.shp'))
    
    WELL_LOC['UWI10'] = WELL_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELL_LOC = WELL_LOC.loc[~(WELL_LOC['UWI10'] == 500000000)]
    WELL_LOC['X'] = WELL_LOC.coords.apply(lambda x:x[0][0])
    WELL_LOC['Y'] = WELL_LOC.coords.apply(lambda x:x[0][1])
    WELL_LOC['XBHL'] = WELL_LOC.coords.apply(lambda x:x[-1][0])
    WELL_LOC['YBHL'] = WELL_LOC.coords.apply(lambda x:x[-1][1])

    WELLPLAN_LOC = read_shapefile(shp.Reader('Directional_Lines_Pending.shp'))
    WELLPLAN_LOC['UWI10'] = WELLPLAN_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLPLAN_LOC = WELLPLAN_LOC.loc[~(WELLPLAN_LOC['UWI10'] == 500000000)]
    WELLPLAN_LOC['X'] = WELLPLAN_LOC.coords.apply(lambda x:x[0][0])
    WELLPLAN_LOC['Y'] = WELLPLAN_LOC.coords.apply(lambda x:x[0][1])
    WELLPLAN_LOC['XBHL'] = WELLPLAN_LOC.coords.apply(lambda x:x[-1][0])
    WELLPLAN_LOC['YBHL'] = WELLPLAN_LOC.coords.apply(lambda x:x[-1][1])

    WELLLINE_LOC = read_shapefile(shp.Reader('Directional_Lines.shp'))
    WELLLINE_LOC['UWI10'] = WELLLINE_LOC.API.apply(lambda x:WELLAPI('05'+str(x)).API2INT(10))
    WELLLINE_LOC = WELLLINE_LOC.loc[~(WELLLINE_LOC['UWI10'] == 500000000)]
    WELLLINE_LOC['X'] = WELLLINE_LOC.coords.apply(lambda x:x[0][0])
    WELLLINE_LOC['Y'] = WELLLINE_LOC.coords.apply(lambda x:x[0][1])
    WELLLINE_LOC['XBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][0])
    WELLLINE_LOC['YBHL'] = WELLLINE_LOC.coords.apply(lambda x:x[-1][1])
    
    LOC_COLS = ['UWI10','X','Y','XBHL','YBHL']
    LOC_DF = WELLLINE_LOC[LOC_COLS].drop_duplicates()
    m = WELLPLAN_LOC.index[~(WELLPLAN_LOC.UWI10.isin(LOC_DF.UWI10))]
    LOC_DF = pd.concat([LOC_DF,WELLPLAN_LOC.loc[m,LOC_COLS].drop_duplicates()])
    m = WELL_LOC.index[~(WELL_LOC.UWI10.isin(LOC_DF.UWI10))]
    LOC_DF = pd.concat([LOC_DF,WELL_LOC.loc[m,LOC_COLS].drop_duplicates()])
    LOC_DF.UWI10.shape[0]-len(LOC_DF.UWI10.unique())
    
    LOC_DF[['XFEET','YFEET']] = pd.DataFrame(convert_XY(LOC_DF.X,LOC_DF.Y,26913,2231)).T.values
    LOC_DF[['XBHLFEET','YBHLFEET']] = pd.DataFrame(convert_XY(LOC_DF.XBHL,LOC_DF.YBHL,26913,2231)).T.values
 
    LOC_DF['DELTA'] = ((LOC_DF['YBHLFEET'] - LOC_DF['YFEET'])**2 +  (LOC_DF['XBHLFEET'] - LOC_DF['XFEET'])**2)**0.5

    if bool(DB_NAME):
        CONN = sqlite3.connect(DB_NAME)
        LOC_COLS = {'UWI10': 'INTEGER',
                    'X': 'REAL',
                    'Y': 'REAL',
                    'XFEET':'REAL',
                    'YFEET':'REAL'}

        INIT_SQL_TABLE(CONN, 'SHL', LOC_COLS)
        LOC_DF[['UWI10','X','Y','XFEET','YFEET']].to_sql(name = 'SHL', con = CONN, if_exists='replace', index = False, dtype = LOC_COLS)
    return LOC_DF
 
