"""Read small members of large remote zip files via HTTP Range requests.

Usage:
  python -I fetch_remote_zip.py list  <url> <out_listing.json>
  python -I fetch_remote_zip.py listnested <url> <outer_listing.json> <member_name> <out_listing.json>
  python -I fetch_remote_zip.py fetch <url> <listing.json> <regex> <out_dir> [threads]

`list` writes the central directory (name, header_offset, compress_size, file_size, method, date).
`listnested` does the same for a zip stored uncompressed as a member of another remote zip.
`fetch` downloads members whose name matches regex (re.search) and writes
<out_dir>/members.json = {name: text_content (utf-8, errors=replace)}.
Standard library only.
"""
import io, json, os, re, struct, sys, time, zlib, zipfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "paiec-ingest/1.0"}


def resolve(url):
    # follow redirects with a 1-byte range GET to get final url + total size
    req = urllib.request.Request(url, headers=dict(UA, Range="bytes=0-0"))
    with urllib.request.urlopen(req, timeout=60) as r:
        final = r.geturl()
        cr = r.headers.get("Content-Range")
        r.read()
    size = int(cr.split("/")[-1])
    if "drive.usercontent.google.com" in url:
        final = url  # signed redirect not used for Drive; keep the original
    return final, size


def get_range(url, start, end, tries=6):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=dict(UA, Range=f"bytes={start}-{end}"))
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
                if r.status != 206 and not (start == 0 and len(data) == end + 1):
                    raise IOError(f"no range support: status {r.status}")
                return data
        except Exception as e:  # noqa
            last = e
            time.sleep(2 * (k + 1))
    raise last


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        # read-ahead for central directory reads: fetch at least 1 MB
        fetch_end = min(max(end, self.pos + (1 << 20) - 1), self.size - 1)
        key = (self.pos, fetch_end)
        data = get_range(self.url, self.pos, fetch_end)
        out = data[: end - self.pos + 1]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def do_list(url, out, base=0, inner_size=None):
    final, size_total = resolve(url)
    size = inner_size if inner_size is not None else size_total
    _get = get_range
    def get_range_b(u, a, b):
        return _get(u, base + a, base + b)
    get_range_local = get_range_b
    # read the tail into memory, large enough for the central directory
    tail_len = min(size, 64 << 20)
    tail = get_range_local(final, size - tail_len, size - 1)

    class TailFile(io.RawIOBase):
        def __init__(s):
            s.pos = 0
        def seekable(s): return True
        def readable(s): return True
        def tell(s): return s.pos
        def seek(s, off, whence=0):
            s.pos = off if whence == 0 else (s.pos + off if whence == 1 else size + off)
            return s.pos
        def read(s, n=-1):
            if n is None or n < 0:
                n = size - s.pos
            lo = s.pos - (size - tail_len)
            if lo < 0:
                d = get_range_local(final, s.pos, min(s.pos + n, size) - 1)
            else:
                d = tail[lo: lo + n]
            s.pos += len(d)
            return d
        def readinto(s, b):
            d = s.read(len(b)); b[: len(d)] = d; return len(d)

    zf = zipfile.ZipFile(io.BufferedReader(TailFile(), buffer_size=1 << 16))
    rows = [dict(name=i.filename, header_offset=i.header_offset, compress_size=i.compress_size,
                 file_size=i.file_size, method=i.compress_type, date=list(i.date_time)) for i in zf.infolist()]
    json.dump(dict(url=url, size=size, base=base, n=len(rows), members=rows), open(out, "w"))
    print(f"{len(rows)} members, size {size}")


def fetch_member(final, m, base=0):
    off = base + m["header_offset"]
    # local header 30 bytes + name + extra (extra unknown, allow 1 KB slack)
    need = 30 + len(m["name"].encode("utf-8")) + 1024 + m["compress_size"]
    d = get_range(final, off, off + need - 1)
    sig, ver, flag, meth, t, dt, crc, cs, us, nl, el = struct.unpack("<IHHHHHIIIHH", d[:30])
    assert sig == 0x04034B50, "bad local header"
    start = 30 + nl + el
    if start + m["compress_size"] > len(d):
        d = d + get_range(final, off + len(d), off + start + m["compress_size"] - 1)
    raw = d[start: start + m["compress_size"]]
    if m["method"] == 0:
        data = raw
    elif m["method"] == 8:
        data = zlib.decompress(raw, -15)
    else:
        raise ValueError(f"method {m['method']}")
    return data.decode("utf-8", errors="replace")


def do_fetch(url, listing, pattern, out_dir, threads=16):
    L = json.load(open(listing))
    rx = re.compile(pattern)
    mem = [m for m in L["members"] if rx.search(m["name"]) and m["file_size"] < (5 << 20)]
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "members.json")
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [m for m in mem if m["name"] not in done]
    final, size = resolve(url)
    t0 = time.time()
    base = L.get("base", 0)
    def job(m):
        return m["name"], fetch_member(final, m, base)
    with ThreadPoolExecutor(threads) as ex:
        for k, (n, txt) in enumerate(ex.map(job, todo)):
            done[n] = txt
            if k % 500 == 0:
                json.dump(done, open(outp, "w"))
    json.dump(done, open(outp, "w"))
    print(f"fetched {len(todo)} (total {len(done)}) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if sys.argv[1] == "list":
        do_list(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "listnested":
        # listnested <url> <outer_listing> <member_name> <out>
        url, outer, member, out = sys.argv[2:6]
        L = json.load(open(outer))
        m = [x for x in L["members"] if x["name"] == member][0]
        assert m["method"] == 0, "inner zip must be stored"
        final, _ = resolve(url)
        hdr = get_range(final, m["header_offset"], m["header_offset"] + 29)
        sig, ver, flag, meth, t, dt, crc, cs, us, nl, el = struct.unpack("<IHHHHHIIIHH", hdr)
        base = m["header_offset"] + 30 + nl + el
        do_list(url, out, base=base, inner_size=m["compress_size"])
    elif sys.argv[1] == "fetch":
        do_fetch(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 16)
