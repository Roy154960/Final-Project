import type { ComponentProps } from "react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8001";

/**
 * react-markdown's own default urlTransform (see its source) only lets
 * "http", "https", "irc", "ircs", "mailto", and "xmpp" URLs through --
 * every other protocol, "data:" included, is silently rewritten to an
 * empty string as an XSS precaution (a clickable `data:text/html` link
 * could run arbitrary script). That's the right call for links, but it
 * was also blanking out the `src` on every image agents/specialists.py's
 * image_qa_node now returns: mcp_server/image_tools.py's
 * retrieve_images_with_data base64-encodes each image into a
 * `data:image/...;base64,...` URI (see that module's own docstring for
 * why -- no static file server needed). With `src=""`, the browser had
 * nothing to paint, so nothing rendered where the image should have
 * been, and the caption text -- both the image's own alt text AND the
 * italic line printed underneath it -- ended up as the only visible
 * output. This is the fix for exactly that.
 *
 * Scoped as narrowly as possible rather than allowing `data:` broadly:
 * only an `<img>` element's `src` attribute (never `href`, never any
 * other tag) with a `data:image/...;base64,...` value passes through
 * unchanged; everything else -- links, non-image data: schemes,
 * javascript:, anything malformed -- still goes through react-markdown's
 * own defaultUrlTransform exactly as before. This is additive, not a
 * relaxation of the existing link-safety behavior.
 */
const DATA_IMAGE_URI_RE = /^data:image\/[a-zA-Z0-9.+-]+;base64,/;

function allowImageDataUris(
  url: string,
  key: string,
  node: { tagName?: string },
): string {
  if (key === "src" && node.tagName === "img" && DATA_IMAGE_URI_RE.test(url)) {
    return url;
  }
  return defaultUrlTransform(url);
}

/**
 * agents/specialists.py's image_qa_node emits image URLs as *relative*
 * paths -- `/images/<filename>` -- served by agents/api.py's own
 * `GET /images/{filename}` endpoint (see that file's docstring for why:
 * `image_path` as returned by retrieve_images() is a path on the
 * server's local disk, unreachable from a browser directly). That path
 * is now only a FALLBACK the node uses when the connected MCP server
 * doesn't expose the newer, base64-embedding `retrieve_images_embedded`
 * tool (see that node's own comments) -- a `data:` src (handled by
 * allowImageDataUris above, not here) is the normal case today.
 *
 * A relative "/images/..." src resolves correctly out of the box for
 * agents/static/chat.html, since that page is served BY the same FastAPI
 * app. This app is different: it's served by Vite on its own origin
 * (see api.ts's API_BASE / VITE_API_BASE_URL), so a bare "/images/..."
 * src would otherwise resolve against Vite's origin instead of the agent
 * API's, and 404. This component is the one place that difference is
 * corrected -- every other relative link (painting_lookup's web source
 * links, for instance) is already a full http(s) URL and passes through
 * untouched. A "data:...;base64,..." src never starts with "/", so it's
 * untouched by this rewrite either way.
 */
function resolveImgSrc(src: string | undefined): string | undefined {
  return src && src.startsWith("/") ? `${API_BASE}${src}` : src;
}

/** data:image/<subtype>;base64,... -> a sane file extension for that subtype. */
const EXTENSION_BY_MIME_SUBTYPE: Record<string, string> = {
  jpeg: "jpg",
  jpg: "jpg",
  png: "png",
  gif: "gif",
  webp: "webp",
  bmp: "bmp",
  tiff: "tiff",
  "svg+xml": "svg",
};

/**
 * Turns a caption into a short, filesystem-safe slug for a downloaded
 * file's name -- e.g. "A wooden palette (mixed oils)!" ->
 * "a-wooden-palette-mixed-oils". Captions are free-form VLM output (see
 * mcp_server/image_tools.py's own note on config.IMAGE_CAPTION_PROMPT),
 * so this can't assume anything about punctuation; non-alphanumeric
 * characters just collapse to hyphens. Falls back to "image" for an
 * empty/entirely-punctuation caption, so a download never ends up named
 * just ".png".
 */
function slugifyCaption(caption: string): string {
  const slug = caption
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
  return slug || "image";
}

/**
 * Picks a sensible downloaded filename from an image's alt text and its
 * (already-resolved) src -- used as the `download` attribute below.
 * Branches on the SAME two src shapes RemoteImage itself ever produces:
 *   - a `data:image/<subtype>;base64,...` URI (the normal case today,
 *     see retrieve_images_embedded) -- extension comes from the MIME
 *     subtype, since there's no filename to read one off of.
 *   - an http(s) URL (the /images/{filename} fallback path, or any
 *     future external image src) -- extension comes off the URL's own
 *     path, same one the server originally served it under.
 * Falls back to ".png" if neither yields a recognizable extension,
 * rather than producing an extensionless file the OS won't know how to
 * open.
 */
function buildDownloadFilename(alt: string, src: string): string {
  const base = slugifyCaption(alt);

  const dataMatch = DATA_IMAGE_URI_RE.exec(src);
  if (dataMatch) {
    const subtype = src.slice(11, src.indexOf(";")).toLowerCase(); // after "data:image/"
    return `${base}.${EXTENSION_BY_MIME_SUBTYPE[subtype] ?? "png"}`;
  }

  try {
    const pathname = new URL(src, window.location.href).pathname;
    const originalName = pathname.split("/").pop() ?? "";
    const dot = originalName.lastIndexOf(".");
    if (dot > 0) {
      return `${base}.${originalName.slice(dot + 1)}`;
    }
  } catch {
    // Malformed/relative-in-a-way-URL can't construct doesn't happen in
    // practice (both src shapes above are always absolute by the time
    // they reach here), but fall through to the plain default rather
    // than letting a download button throw.
  }

  return `${base}.png`;
}

/**
 * Small circular download button overlaid on an image's corner -- lets
 * someone save a corpus image straight out of the chat. Works for both
 * src shapes unchanged: a `data:` URI downloads directly from what's
 * already in the page (no network request at all -- the browser just
 * writes the bytes it already has to disk), and an http(s) `/images/...`
 * URL downloads via a normal fetch the `download` attribute triggers.
 * Purely client-side -- no backend endpoint needed for either case.
 */
function ImageDownloadButton({ src, alt }: { src: string; alt: string }) {
  return (
    <a
      href={src}
      download={buildDownloadFilename(alt, src)}
      title={`Download ${alt || "image"}`}
      aria-label={`Download ${alt || "image"}`}
      onClick={(e) => e.stopPropagation()}
      style={{
        position: "absolute",
        top: 8,
        right: 8,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 28,
        height: 28,
        borderRadius: "50%",
        background: "rgba(0, 0, 0, 0.55)",
        color: "#fff",
        lineHeight: 0,
      }}
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M12 3v12" />
        <path d="M7 10l5 5 5-5" />
        <path d="M5 21h14" />
      </svg>
    </a>
  );
}

function RemoteImage({ node, ...rest }: ComponentProps<"img"> & { node?: unknown }) {
  // react-markdown passes an extra `node` prop into every custom
  // component (the hast AST node it rendered from, for components that
  // want to inspect source position/metadata -- see react-markdown's
  // own `passNode: true`). Destructured out and discarded here rather
  // than included in `rest`: spreading it onto a native DOM element
  // like the original version of this component did stringifies the
  // object into an invalid `node="[object Object]"` HTML attribute --
  // harmless to rendering, but not a real attribute, so there's no
  // reason to leave it on the element.
  const src = resolveImgSrc(rest.src);
  const alt = rest.alt ?? "";

  if (!src) {
    // No src at all (couldn't resolve) -- render exactly what the old
    // code did: a plain <img>, which the browser degrades to alt text.
    // Nothing to attach a download button to.
    return <img {...rest} alt={alt} />;
  }

  return (
    <span style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
      <img
        {...rest}
        src={src}
        loading="lazy"
        style={{ maxWidth: "100%", maxHeight: 360, borderRadius: 8, display: "block" }}
      />
      <ImageDownloadButton src={src} alt={alt} />
    </span>
  );
}

/**
 * Plugged into AssistantMessage (see Message.tsx) as MessagePrimitive.
 * Content's `Text` component, so every assistant text part -- not just
 * image_qa's -- renders as real markdown instead of literal
 * "![caption](url)" characters. Deliberately NOT applied to
 * UserMessage: what a person types is shown verbatim, never
 * markdown-interpreted.
 */
export function MarkdownText() {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm]}
      urlTransform={allowImageDataUris}
      className="aui-md"
      components={{ img: RemoteImage }}
    />
  );
}
