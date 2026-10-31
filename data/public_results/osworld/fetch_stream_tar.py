"""Stream a remote tar.gz (optionally stored as a deflated member inside a zip) and keep small files.

Usage:
  python -I fetch_stream_tar.py targz <url> <out_dir> <regex>
  python -I fetch_stream_tar.py zipmember <url> <listing.json> <member_name> <out_dir> <regex>
Writes <out_dir>/members.json = {name: text} and <out_dir>/tar_index.json = [[name, size, mtime], ...].
Standard library only. Never extracts to disk.
"""
import io, json, os, re, struct, sys, tarfile, time, urllib.request, zlib

UA = {"User-Agent": "paiec-ingest/1.0"}


class HTTPStream(io.RawIOBase):
    """Sequential reader over a byte range, reconnecting on errors."""

    def __init__(self, url, start=0, end=None):
        self.url, self.pos, self.end = url, start, end
        self.resp = None
        self.t0 = time.time(); self.nread = 0

    def _open(self):
        rng = f"bytes={self.pos}-" + ("" if self.end is None else str(self.end))
        req = urllib.request.Request(self.url, headers=dict(UA, Range=rng))
        self.resp = urllib.request.urlopen(req, timeout=120)

    def readable(self):
        return True

    def readinto(self, b):
        if self.end is not None and self.pos > self.end:
            return 0
        for k in range(8):
            try:
                if self.resp is None:
                    self._open()
                n = self.resp.readinto(b)
                if n == 0 and len(b) and self.end is not None and self.pos <= self.end:
                    raise IOError(f"connection closed early at byte {self.pos}")   # http.client returns 0 here, not an error
                self.pos += n
                self.nread += n
                if self.nread and self.nread % (256 << 20) < n:
                    print(f"  read {self.nread/1e9:.2f} GB in {time.time()-self.t0:.0f}s", flush=True)
                return n
            except Exception as e:
                print("  reconnect:", repr(e)[:200], flush=True)
                self.resp = None
                time.sleep(3 * (k + 1))
        raise IOError("too many reconnects")


class InflateStream(io.RawIOBase):
    def __init__(self, raw):
        self.raw = raw
        self.d = zlib.decompressobj(-15)
        self.buf = b""
        self.eof = False

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf and not self.eof:
            chunk = self.raw.read(1 << 20)
            if not chunk:
                self.buf += self.d.flush(); self.eof = True
                break
            self.buf += self.d.decompress(chunk)
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


def resolve(url):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl(), int(r.headers.get("Content-Length"))


def scan(fileobj, out_dir, rx):
    os.makedirs(out_dir, exist_ok=True)
    keep, index = {}, []
    tf = tarfile.open(fileobj=fileobj, mode="r|gz")
    for ti in tf:
        index.append([ti.name, ti.size, ti.mtime])
        if ti.isfile() and rx.search(ti.name) and ti.size < (5 << 20):
            f = tf.extractfile(ti)
            keep[ti.name] = f.read().decode("utf-8", errors="replace")
    json.dump(keep, open(os.path.join(out_dir, "members.json"), "w"))
    json.dump(index, open(os.path.join(out_dir, "tar_index.json"), "w"))
    print(f"kept {len(keep)} of {len(index)} tar entries")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "targz":
        url, out_dir, pat = sys.argv[2:5]
        final, size = resolve(url)
        raw = io.BufferedReader(HTTPStream(final, 0, size - 1), buffer_size=1 << 20)
        scan(raw, out_dir, re.compile(pat))
    elif mode == "zipmember":
        url, listing, member, out_dir, pat = sys.argv[2:7]
        L = json.load(open(listing))
        m = [x for x in L["members"] if x["name"] == member][0]
        final, size = resolve(url)
        hdr = urllib.request.urlopen(urllib.request.Request(final, headers=dict(UA, Range=f"bytes={m['header_offset']}-{m['header_offset']+29}")), timeout=60).read()
        sig, ver, flag, meth, t, dt, crc, cs, us, nl, el = struct.unpack("<IHHHHHIIIHH", hdr)
        assert sig == 0x04034B50
        start = m["header_offset"] + 30 + nl + el
        end = start + m["compress_size"] - 1
        raw = io.BufferedReader(HTTPStream(final, start, end), buffer_size=1 << 20)
        if m["method"] == 8:
            raw = io.BufferedReader(InflateStream(raw), buffer_size=1 << 20)
        scan(raw, out_dir, re.compile(pat))
