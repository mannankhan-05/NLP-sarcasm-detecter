import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  History,
  Info,
  Moon,
  Sun,
  Sparkles,
  Upload,
  Video,
  Zap,
} from "lucide-react";

const TABS = [
  { id: "text", label: "Text" },
  { id: "upload", label: "Audio / Video" },
  { id: "models", label: "Model info" },
];

function api(path, options) {
  return fetch(path, options).then(async (res) => {
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || `Request failed (${res.status})`);
    return data;
  });
}

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("mustard-theme") || "dark");
  const [tab, setTab] = useState("text");
  const [text, setText] = useState("");
  const [context, setContext] = useState("");
  const [examples, setExamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [multiResult, setMultiResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("mustard-history") || "[]");
    } catch {
      return [];
    }
  });
  const [fileName, setFileName] = useState("");
  const [transcript, setTranscript] = useState("");
  const [demos, setDemos] = useState([]);
  const dropRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("mustard-theme", theme);
  }, [theme]);

  useEffect(() => {
    api("/examples").then((d) => setExamples(d.items || [])).catch(() => {});
    api("/examples/demo").then((d) => setDemos(d.items || [])).catch(() => {});
    api("/model/metrics").then(setMetrics).catch(() => {});
  }, []);

  useEffect(() => {
    localStorage.setItem("mustard-history", JSON.stringify(history.slice(0, 20)));
  }, [history]);

  const analyzeText = async (value = text, ctx = context) => {
    if (!value.trim()) return;
    setLoading(true);
    setError("");
    try {
      const data = await api("/predict/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: value, context: ctx }),
      });
      setResult(data);
      setHistory((h) => [{ ...data, input: value, at: Date.now(), kind: "text" }, ...h].slice(0, 20));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const analyzeFile = async (file, forcedTranscript = null) => {
    if (!file) return;
    setFileName(file.name);
    setLoading(true);
    setError("");
    const spoken = (forcedTranscript ?? transcript).trim();
    const form = new FormData();
    form.append("video", file);
    if (spoken) form.append("transcript", spoken);
    if (context.trim()) form.append("context", context.trim());
    try {
      const data = await api("/predict/multimodal", { method: "POST", body: form });
      setMultiResult(data);
      setHistory((h) => [{ ...data, input: file.name, at: Date.now(), kind: "clip" }, ...h].slice(0, 20));
      if (data.transcript) setTranscript(data.transcript);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const testDemo = async (demo) => {
    setTab("upload");
    if (demo.transcript) setTranscript(demo.transcript);
    setLoading(true);
    setError("");
    try {
      const res = await fetch(demo.url);
      if (!res.ok) throw new Error(`Could not download ${demo.file}`);
      const blob = await res.blob();
      const file = new File([blob], demo.file, { type: blob.type || (demo.kind === "audio" ? "audio/wav" : "video/mp4") });
      await analyzeFile(file, demo.transcript || "");
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-mustard-400 focus:px-3 focus:py-2 focus:text-ink-950"
      >
        Skip to analysis
      </a>
      <header className="sticky top-0 z-20 border-b border-black/5 bg-mustard-50/80 backdrop-blur dark:border-white/10 dark:bg-ink-950/80">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3">
          <div className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-2xl bg-mustard-400 text-ink-950 shadow-card" aria-hidden>
              <Zap size={18} />
            </span>
            <div>
              <p className="font-display text-xl leading-none">MUSTARD</p>
              <p className="text-xs text-ink-950/60 dark:text-mustard-50/60">
                Multimodal sarcasm detection
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="rounded-full border border-black/10 p-2 dark:border-white/10"
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </div>
      </header>

      <main id="main" className="mx-auto grid max-w-6xl gap-6 px-4 py-8 lg:grid-cols-[1.15fr_0.85fr]">
        <section className="space-y-5">
          <div>
            <p className="text-sm uppercase tracking-[0.18em] text-mustard-600 dark:text-mustard-400">
              Live inference
            </p>
            <h1 className="mt-1 font-display text-4xl leading-tight md:text-5xl">
              The gap between what is said and what is meant.
            </h1>
            <p className="mt-3 max-w-xl text-sm text-ink-950/70 dark:text-mustard-50/70">
              Frozen DistilRoBERTa, audio, and visual encoders with a gated fusion head.
              Type a line, or upload a clip. Text-only runs stay honest: audio and vision are masked, not invented.
            </p>
          </div>

          <nav aria-label="Analysis mode" className="flex flex-wrap gap-2">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? "page" : undefined}
                className={`rounded-full px-4 py-2 text-sm transition ${
                  tab === t.id
                    ? "bg-ink-950 text-mustard-50 dark:bg-mustard-400 dark:text-ink-950"
                    : "bg-white/70 text-ink-950/70 hover:bg-white dark:bg-ink-800 dark:text-mustard-50/70"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>

          {tab === "text" && (
            <div className="rounded-3xl bg-white p-5 shadow-card dark:bg-ink-900">
              <label htmlFor="utterance" className="text-sm font-medium">
                Utterance
              </label>
              <textarea
                id="utterance"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={5}
                placeholder="Oh great, another meeting that could have been an email"
                className="mt-2 w-full resize-y rounded-2xl border border-black/10 bg-mustard-50/60 px-4 py-3 text-base outline-none ring-mustard-400 focus:ring-2 dark:border-white/10 dark:bg-ink-800"
              />
              <label htmlFor="context" className="mt-4 block text-sm font-medium">
                Optional dialogue context
              </label>
              <input
                id="context"
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="What was said just before this line (recommended — isolated text is weaker)"
                className="mt-2 w-full rounded-2xl border border-black/10 bg-mustard-50/60 px-4 py-3 outline-none ring-mustard-400 focus:ring-2 dark:border-white/10 dark:bg-ink-800"
              />
              <div className="mt-4 flex flex-wrap gap-2" aria-label="Example inputs">
                {examples.map((ex) => (
                  <button
                    key={ex.title}
                    type="button"
                    onClick={() => {
                      setText(ex.text);
                      setContext(ex.context || "");
                      analyzeText(ex.text, ex.context || "");
                    }}
                    className="rounded-full border border-black/10 px-3 py-1 text-xs hover:border-mustard-500 dark:border-white/10"
                  >
                    {ex.title}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => analyzeText()}
                disabled={loading || !text.trim()}
                className="mt-5 w-full rounded-2xl bg-mustard-400 py-3 font-medium text-ink-950 transition hover:bg-mustard-500 disabled:opacity-50"
              >
                {loading ? "Analyzing…" : "Analyze"}
              </button>
            </div>
          )}

          {tab === "upload" && (
            <div className="rounded-3xl bg-white p-5 shadow-card dark:bg-ink-900">
              <div
                ref={dropRef}
                role="button"
                tabIndex={0}
                aria-label="Upload a video or audio clip"
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") dropRef.current?.querySelector("input")?.click();
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const f = e.dataTransfer.files?.[0];
                  if (f) analyzeFile(f, "");
                }}
                className="grid place-items-center rounded-2xl border border-dashed border-mustard-500/50 bg-mustard-50/50 px-4 py-10 text-center dark:bg-ink-800"
              >
                <Upload className="mb-2" />
                <p className="font-medium">Drop a clip, or choose a file</p>
                <p className="mt-1 text-xs text-ink-950/60 dark:text-mustard-50/60">
                  mp4, mov, webm, wav — 80 MB max
                </p>
                <input
                  className="mt-4 text-sm"
                  type="file"
                  accept="video/*,audio/*"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) analyzeFile(f, "");
                  }}
                />
                {fileName && <p className="mt-2 text-xs">Selected: {fileName}</p>}
              </div>
              <label htmlFor="transcript" className="mt-4 block text-sm font-medium">
                Transcript (recommended)
              </label>
              <textarea
                id="transcript"
                value={transcript}
                onChange={(e) => setTranscript(e.target.value)}
                rows={3}
                placeholder="Leave blank to auto-transcribe your recording, or paste what was said."
                className="mt-2 w-full rounded-2xl border border-black/10 bg-mustard-50/60 px-4 py-3 outline-none ring-mustard-400 focus:ring-2 dark:border-white/10 dark:bg-ink-800"
              />
              {demos.length > 0 && (
                <div className="mt-5 space-y-3">
                  <h2 className="text-sm font-medium">Labeled demo clips</h2>
                  <p className="text-xs text-ink-950/60 dark:text-mustard-50/60">
                    Supported: MP4 (H.264+AAC) and WAV 16 kHz. The 50/50 result on silent/unlabeled clips is gone — these files have speech plus an embedded transcript.
                  </p>
                  <ul className="space-y-2">
                    {demos.map((demo) => (
                      <li
                        key={demo.id || demo.file}
                        className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-black/10 px-3 py-2 dark:border-white/10"
                      >
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium">{demo.file}</p>
                          <p className="text-[11px] text-ink-950/60 dark:text-mustard-50/60">
                            {demo.format} · {demo.seconds}s · expected{" "}
                            <span className="font-semibold uppercase">{String(demo.expected).replace("_", "-")}</span>
                          </p>
                          <p className="mt-0.5 line-clamp-2 text-[11px] italic opacity-70">“{demo.transcript}”</p>
                        </div>
                        <div className="flex gap-2">
                          <a
                            href={demo.url}
                            download={demo.file}
                            className="rounded-full border border-black/10 px-3 py-1 text-xs dark:border-white/10"
                          >
                            Download
                          </a>
                          <button
                            type="button"
                            onClick={() => testDemo(demo)}
                            className="rounded-full bg-mustard-400 px-3 py-1 text-xs font-medium text-ink-950"
                          >
                            Test
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {tab === "models" && <ModelInfo metrics={metrics} />}

          {error && (
            <div role="alert" className="flex items-start gap-2 rounded-2xl bg-red-100 px-4 py-3 text-sm text-red-900 dark:bg-red-950/60 dark:text-red-100">
              <AlertCircle size={16} className="mt-0.5 shrink-0" />
              <span>{typeof error === "string" ? error : "Something went wrong."}</span>
            </div>
          )}
        </section>

        <aside className="space-y-5">
          {loading && <Skeleton />}
          {!loading && tab !== "models" && (
            <ResultsPanel
              result={tab === "upload" ? multiResult : result}
              multimodal={tab === "upload"}
            />
          )}
          <HistoryList items={history} onPick={(item) => setResult(item)} />
        </aside>
      </main>
    </div>
  );
}

function ResultsPanel({ result, multimodal }) {
  if (!result) {
    return (
      <div className="rounded-3xl border border-dashed border-black/10 p-6 text-sm text-ink-950/60 dark:border-white/10 dark:text-mustard-50/60">
        Run an analysis to see the verdict, calibrated confidence, token attributions, and which modalities were actually used.
      </div>
    );
  }
  const sarcastic = result.label === "sarcastic";
  return (
    <div className="space-y-4">
      <div className="rounded-3xl bg-white p-5 shadow-card dark:bg-ink-900">
        <div className="flex items-start justify-between gap-4">
          <VerdictBadge sarcastic={sarcastic} />
          <Gauge value={result.confidence} />
        </div>
        <dl className="mt-5 grid grid-cols-2 gap-3 text-sm">
          <div>
            <dt className="text-ink-950/50 dark:text-mustard-50/50">Non-sarcastic</dt>
            <dd className="font-medium">{pct(result.probabilities?.non_sarcastic)}</dd>
            <Bar value={result.probabilities?.non_sarcastic} tone="slate" />
          </div>
          <div>
            <dt className="text-ink-950/50 dark:text-mustard-50/50">Sarcastic</dt>
            <dd className="font-medium">{pct(result.probabilities?.sarcastic)}</dd>
            <Bar value={result.probabilities?.sarcastic} tone="mustard" />
          </div>
        </dl>
        <ModalityChips used={result.modalities_used} multimodal={multimodal} />
        {multimodal && result.transcript && (
          <p className="mt-3 text-sm">
            <span className="text-xs uppercase tracking-wide text-ink-950/50 dark:text-mustard-50/50">
              {result.transcript_source === "asr" ? "Heard (auto-transcribed)" : "Transcript"}
            </span>
            <span className="mt-1 block italic">“{result.transcript}”</span>
          </p>
        )}
        {result.note && (
          <p className="mt-3 text-xs text-ink-950/60 dark:text-mustard-50/60">{result.note}</p>
        )}
        <p className="mt-2 text-[11px] text-ink-950/40 dark:text-mustard-50/40">
          {result.model} · {result.inference_ms} ms
        </p>
      </div>

      {result.token_attributions?.length > 0 && (
        <div className="rounded-3xl bg-white p-5 shadow-card dark:bg-ink-900">
          <h2 className="font-display text-lg">Words driving the decision</h2>
          <p className="mb-3 text-xs text-ink-950/60 dark:text-mustard-50/60">
            Shade intensity is attribution magnitude, not sentiment. Meaning is not carried by color alone — scores are in the tooltip.
          </p>
          <TokenCloud tokens={result.token_attributions} />
        </div>
      )}

      {result.modality_contributions && multimodal && (
        <div className="rounded-3xl bg-white p-5 shadow-card dark:bg-ink-900">
          <h2 className="font-display text-lg">Per-modality contribution</h2>
          {Object.entries(result.modality_contributions).map(([k, v]) => (
            <div key={k} className="mt-2">
              <div className="flex justify-between text-xs uppercase tracking-wide">
                <span>{k}</span>
                <span>{pct(v)}</span>
              </div>
              <Bar value={v} tone="mustard" />
            </div>
          ))}
        </div>
      )}

      {result.keyframes?.length > 0 && (
        <div className="rounded-3xl bg-white p-5 shadow-card dark:bg-ink-900">
          <h2 className="mb-3 flex items-center gap-2 font-display text-lg">
            <Video size={18} /> Keyframes
          </h2>
          <div className="grid grid-cols-3 gap-2">
            {result.keyframes.map((k) => (
              <figure key={k.gradcam_url}>
                <img src={k.gradcam_url} alt={`Frame at ${k.timestamp}s`} className="h-24 w-full rounded-xl object-cover" />
                <figcaption className="mt-1 text-[11px]">{k.timestamp}s</figcaption>
              </figure>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function VerdictBadge({ sarcastic }) {
  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-medium ${
        sarcastic
          ? "bg-mustard-100 text-mustard-600 dark:bg-mustard-400/15 dark:text-mustard-400"
          : "bg-slate-200 text-slate-700 dark:bg-slate-700/40 dark:text-slate-200"
      }`}
    >
      {sarcastic ? <Sparkles size={16} aria-hidden /> : <CheckCircle2 size={16} aria-hidden />}
      <span>{sarcastic ? "SARCASTIC" : "NON-SARCASTIC"}</span>
    </div>
  );
}

function Gauge({ value = 0 }) {
  const pctVal = Math.round((value || 0) * 100);
  const r = 36;
  const c = 2 * Math.PI * r;
  const offset = c - (pctVal / 100) * c;
  return (
    <div className="relative h-24 w-24" role="img" aria-label={`Confidence ${pctVal} percent`}>
      <svg viewBox="0 0 88 88" className="-rotate-90">
        <circle cx="44" cy="44" r={r} fill="none" stroke="currentColor" strokeWidth="8" className="opacity-15" />
        <circle
          cx="44"
          cy="44"
          r={r}
          fill="none"
          stroke="#C9961A"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          className="gauge-arc"
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <span className="text-lg font-semibold leading-none">{pctVal}%</span>
      </div>
    </div>
  );
}

function Bar({ value = 0, tone }) {
  return (
    <div className="mt-1 h-2 overflow-hidden rounded-full bg-black/10 dark:bg-white/10" aria-hidden>
      <div
        className={`h-full rounded-full ${tone === "mustard" ? "bg-mustard-500" : "bg-slate-500"}`}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
      />
    </div>
  );
}

function ModalityChips({ used = [], multimodal }) {
  const all = ["text", "audio", "visual"];
  return (
    <div className="mt-4">
      <p className="text-xs uppercase tracking-wide text-ink-950/50 dark:text-mustard-50/50">Modalities used</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {all.map((m) => {
          const on = used.includes(m);
          return (
            <span
              key={m}
              className={`rounded-full px-3 py-1 text-xs ${
                on
                  ? "bg-mustard-400 text-ink-950"
                  : "bg-black/5 text-ink-950/50 dark:bg-white/10 dark:text-mustard-50/50"
              }`}
            >
              {on ? "●" : "○"} {m}
            </span>
          );
        })}
      </div>
      {!multimodal && (
        <p className="mt-2 text-xs">Text only — upload a clip for full multimodal analysis.</p>
      )}
    </div>
  );
}

function TokenCloud({ tokens }) {
  return (
    <p className="flex flex-wrap gap-1 text-base leading-relaxed">
      {tokens.map((t, i) => {
        const mag = Math.min(1, Math.abs(t.score));
        const bg = `rgba(201, 150, 26, ${0.12 + mag * 0.55})`;
        return (
          <span
            key={`${t.token}-${i}`}
            title={`${t.token}: ${t.score}`}
            className="rounded-md px-1"
            style={{ background: bg, fontWeight: mag > 0.6 ? 650 : 460 }}
          >
            {t.token}
          </span>
        );
      })}
    </p>
  );
}

function Skeleton() {
  return (
    <div className="animate-pulse space-y-3 rounded-3xl bg-white p-5 dark:bg-ink-900" aria-live="polite" aria-busy="true">
      <div className="h-8 w-40 rounded-full bg-black/10 dark:bg-white/10" />
      <div className="h-24 rounded-2xl bg-black/10 dark:bg-white/10" />
      <div className="h-4 w-2/3 rounded bg-black/10 dark:bg-white/10" />
      <span className="sr-only">Analyzing utterance</span>
    </div>
  );
}

function HistoryList({ items, onPick }) {
  if (!items.length) return null;
  return (
    <div className="rounded-3xl bg-white p-5 shadow-card dark:bg-ink-900">
      <h2 className="mb-3 flex items-center gap-2 font-display text-lg">
        <History size={16} /> Session history
      </h2>
      <ul className="space-y-2 text-sm">
        {items.slice(0, 8).map((it, i) => (
          <li key={`${it.at}-${i}`}>
            <button
              type="button"
              onClick={() => onPick(it)}
              className="w-full rounded-xl px-2 py-2 text-left hover:bg-mustard-50 dark:hover:bg-ink-800"
            >
              <span className="block truncate">{it.input}</span>
              <span className="text-xs opacity-60">
                {it.label} · {pct(it.confidence)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ModelInfo({ metrics }) {
  if (!metrics) {
    return (
      <div className="rounded-3xl bg-white p-5 text-sm shadow-card dark:bg-ink-900">
        Metrics appear after `python train.py && python evaluate.py`.
      </div>
    );
  }
  const cm = metrics.confusion_matrix || [[0, 0], [0, 0]];
  const ablation = metrics.ablation_table || {};
  return (
    <div className="space-y-4 rounded-3xl bg-white p-5 shadow-card dark:bg-ink-900">
      <h2 className="flex items-center gap-2 font-display text-xl">
        <Info size={18} /> Test-set protocol
      </h2>
      <p className="text-sm opacity-70">{metrics.protocol}</p>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <Stat label="Accuracy" value={fmt(metrics.accuracy, metrics.accuracy_std)} />
        <Stat label="Macro-F1" value={fmt(metrics.macro_f1, metrics.macro_f1_std)} />
        <Stat label="Precision" value={fmt(metrics.precision)} />
        <Stat label="Recall" value={fmt(metrics.recall)} />
        <Stat label="ECE" value={fmt(metrics.ece)} />
        <Stat label="ROC-AUC" value={fmt(metrics.roc_auc)} />
      </div>
      <div>
        <p className="mb-2 text-sm font-medium">Confusion matrix (summed folds)</p>
        <table className="w-full text-center text-sm">
          <thead>
            <tr>
              <th />
              <th>Pred NS</th>
              <th>Pred S</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th className="text-left">True NS</th>
              <td>{cm[0]?.[0]}</td>
              <td>{cm[0]?.[1]}</td>
            </tr>
            <tr>
              <th className="text-left">True S</th>
              <td>{cm[1]?.[0]}</td>
              <td>{cm[1]?.[1]}</td>
            </tr>
          </tbody>
        </table>
      </div>
      {Object.keys(ablation).length > 0 && (
        <div>
          <p className="mb-2 text-sm font-medium">Modality ablation (macro-F1)</p>
          <ul className="space-y-1 text-sm">
            {Object.entries(ablation).map(([k, v]) => (
              <li key={k} className="flex justify-between">
                <span>{k}</span>
                <span>{fmt(v.macro_f1, v.macro_f1_std)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-2xl bg-mustard-50 px-3 py-2 dark:bg-ink-800">
      <div className="text-[11px] uppercase tracking-wide opacity-60">{label}</div>
      <div className="font-medium">{value}</div>
    </div>
  );
}

function pct(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

function fmt(mean, std) {
  if (mean == null || Number.isNaN(mean)) return "—";
  const m = Number(mean).toFixed(3);
  return std == null || Number.isNaN(std) ? m : `${m} ± ${Number(std).toFixed(3)}`;
}
