import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BatteryCharging,
  Car,
  CheckCircle2,
  Cpu,
  Gauge,
  Info,
  Loader2,
  Sparkles,
  Tag,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { fetchModelInfo, fetchOptions, predictCarPrice } from "./api";

function formatPKRShort(amount) {
  if (!amount && amount !== 0) return "";
  if (amount >= 10_000_000) {
    return `${(amount / 10_000_000).toFixed(2)} Crore`;
  }
  if (amount >= 100_000) {
    return `${(amount / 100_000).toFixed(2)} Lacs`;
  }
  return `${amount.toLocaleString()} PKR`;
}

const TEAM_OPTIONS = [
  {
    id: "faiz",
    name: "Faiz",
    algo: "Random Forest",
    tag: "Best R² (0.932)",
  },
  {
    id: "farhan",
    name: "Farhan",
    algo: "Decision Tree",
    tag: "R² 0.921",
  },
  {
    id: "ahmed",
    name: "Ahmed",
    algo: "KNN (k=5 / k=7)",
    tag: "Best Clf (92.3%)",
  },
];

const BAND_STYLES = {
  Budget: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  "Mid-range": "bg-indigo-500/15 text-indigo-300 border-indigo-500/30",
  Premium: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  Luxury: "bg-purple-500/15 text-purple-300 border-purple-500/30",
};

const VERDICT_CONFIG = {
  fair: {
    label: "Fair Market Price",
    icon: CheckCircle2,
    card: "bg-fuchsia-500/10 border-fuchsia-500/30 text-fuchsia-200",
    badge: "bg-fuchsia-500/20 text-fuchsia-300",
  },
  possible_deal: {
    label: "Possible Good Deal",
    icon: TrendingDown,
    card: "bg-cyan-500/10 border-cyan-500/30 text-cyan-200",
    badge: "bg-cyan-500/20 text-cyan-300",
  },
  overpriced: {
    label: "Above Market Estimate",
    icon: TrendingUp,
    card: "bg-rose-500/10 border-rose-500/30 text-rose-200",
    badge: "bg-rose-500/20 text-rose-300",
  },
};

export default function App() {
  const [options, setOptions] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [bootError, setBootError] = useState("");

  // Selected teammate model
  const [author, setAuthor] = useState("faiz");

  // Form state
  const [make, setMake] = useState("Toyota");
  const [model, setModel] = useState("Corolla");
  const [variant, setVariant] = useState("GLi 1.3");
  const [modelYear, setModelYear] = useState(2020);
  const [mileage, setMileage] = useState(55000);
  const [fuelType, setFuelType] = useState("Petrol");
  const [transmission, setTransmission] = useState("Automatic");
  const [engineCc, setEngineCc] = useState(1300);
  const [batteryKwh, setBatteryKwh] = useState(60);
  const [auctionRating, setAuctionRating] = useState("");
  const [listedPrice, setListedPrice] = useState("3800000");

  // Prediction state
  const [loading, setLoading] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    Promise.all([fetchOptions(), fetchModelInfo()])
      .then(([opts, info]) => {
        setOptions(opts);
        setModelInfo(info);
      })
      .catch((err) => setBootError(err.message));
  }, []);

  const availableModels = useMemo(() => {
    if (!options?.catalog?.[make]) return [];
    return Object.keys(options.catalog[make]).sort();
  }, [options, make]);

  const availableVariants = useMemo(() => {
    return options?.catalog?.[make]?.[model]?.variants || [];
  }, [options, make, model]);

  const handleMakeChange = (newMake) => {
    setMake(newMake);
    const models = Object.keys(options?.catalog?.[newMake] || {}).sort();
    const nextModel = models[0] || "";
    setModel(nextModel);

    const entry = options?.catalog?.[newMake]?.[nextModel];
    setVariant(entry?.variants?.[0] || "");
    if (entry?.typical_engine_cc === null) {
      setFuelType("Electric");
    } else {
      if (fuelType === "Electric") setFuelType("Petrol");
      if (entry?.typical_engine_cc) setEngineCc(entry.typical_engine_cc);
    }
  };

  const handleModelChange = (newModel) => {
    setModel(newModel);
    const entry = options?.catalog?.[make]?.[newModel];
    setVariant(entry?.variants?.[0] || "");
    if (entry?.typical_engine_cc === null) {
      setFuelType("Electric");
    } else {
      if (fuelType === "Electric") setFuelType("Petrol");
      if (entry?.typical_engine_cc) setEngineCc(entry.typical_engine_cc);
    }
  };

  const isElectric = fuelType === "Electric";

  const runPrediction = async (selectedAuthor = author) => {
    setSubmitError("");
    setLoading(true);
    try {
      const payload = {
        author: selectedAuthor,
        make,
        model,
        variant,
        model_year: Number(modelYear),
        mileage: Number(mileage),
        fuel_type: fuelType,
        transmission,
        engine_cc: isElectric ? null : Number(engineCc),
        battery_kwh: isElectric ? Number(batteryKwh) : null,
        auction_rating: auctionRating === "" ? null : Number(auctionRating),
        listed_price_pkr: listedPrice === "" ? null : Number(listedPrice),
      };
      const data = await predictCarPrice(payload);
      setResult(data);
    } catch (err) {
      setSubmitError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await runPrediction(author);
  };

  const handleAuthorSwitch = async (nextAuthor) => {
    setAuthor(nextAuthor);
    if (result && options) {
      await runPrediction(nextAuthor);
    }
  };

  const activeBenchmarks = useMemo(() => {
    const rows = modelInfo?.team?.[author]?.metrics;
    if (!Array.isArray(rows)) return null;
    const testRows = rows.filter((r) => r.set === "test");
    const bestReg = testRows
      .filter((r) => r.task === "regression" && r.MAE != null)
      .sort((a, b) => a.MAE - b.MAE)[0];
    const bestClf = testRows
      .filter((r) => r.task === "classification" && r.Macro_F1 != null)
      .sort((a, b) => b.Macro_F1 - a.Macro_F1)[0];
    return { bestReg, bestClf };
  }, [modelInfo, author]);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 selection:bg-fuchsia-500/30">
      {/* Top Header */}
      <header className="sticky top-0 z-20 border-b border-neutral-800/80 bg-neutral-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] w-full items-center justify-between px-4 py-3.5 sm:px-8">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-fuchsia-500/15 text-fuchsia-400 ring-1 ring-fuchsia-500/30">
              <Car className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white">
                AutoPulse <span className="text-fuchsia-400">PK</span>
              </h1>
              <p className="text-xs text-neutral-400">
                Used Car Fair-Value & Deal Analyzer
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-neutral-800 bg-neutral-900 px-3 py-1 text-xs text-neutral-300">
            <span
              className={`h-2 w-2 rounded-full ${
                options ? "bg-fuchsia-400 animate-pulse" : "bg-amber-400"
              }`}
            />
            {options ? `${options.makes.length} Brands • 3 Team Models` : "Connecting..."}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] w-full px-4 py-6 sm:px-8 lg:py-8">
        {bootError && (
          <div className="mb-6 flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-rose-400" />
            <div>
              <p className="font-semibold">Backend API Unreachable</p>
              <p className="mt-0.5 text-rose-300/90">
                {bootError}. Make sure FastAPI is running on port 8000:{" "}
                <code className="rounded bg-neutral-900 px-1.5 py-0.5 text-xs">
                  uvicorn backend.main:app --host 0.0.0.0 --reload --port 8000
                </code>
              </p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 lg:items-stretch">
          {/* LEFT COLUMN: Input Form */}
          <form
            onSubmit={handleSubmit}
            className="space-y-5 rounded-2xl border border-neutral-800 bg-neutral-900/70 p-5 shadow-xl sm:p-6 lg:col-span-7"
          >
            {/* Team Model Switcher */}
            <div className="rounded-xl border border-neutral-800 bg-neutral-950/80 p-3.5">
              <div className="mb-2 flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-xs font-semibold text-neutral-200">
                  <Cpu className="h-3.5 w-3.5 text-fuchsia-400" />
                  Select Team Model Pipeline
                </span>
                <span className="text-[11px] text-neutral-400">
                  Switch anytime to compare
                </span>
              </div>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                {TEAM_OPTIONS.map((t) => {
                  const active = author === t.id;
                  return (
                    <button
                      type="button"
                      key={t.id}
                      onClick={() => handleAuthorSwitch(t.id)}
                      className={`flex flex-col items-start rounded-xl border px-3 py-2 text-left transition ${
                        active
                          ? "border-fuchsia-500 bg-fuchsia-500/15 text-white shadow-sm shadow-fuchsia-500/10"
                          : "border-neutral-800 bg-neutral-900/90 text-neutral-400 hover:border-neutral-700 hover:text-neutral-200"
                      }`}
                    >
                      <div className="flex w-full items-center justify-between">
                        <span className="text-xs font-bold">{t.name}</span>
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            active
                              ? "bg-fuchsia-500/25 text-fuchsia-300"
                              : "bg-neutral-800 text-neutral-400"
                          }`}
                        >
                          {t.tag}
                        </span>
                      </div>
                      <span className="mt-0.5 text-[11px] opacity-80">
                        {t.algo}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
              <h2 className="text-base font-semibold text-white">
                Vehicle Specifications
              </h2>
              <span className="text-xs text-neutral-400">
                PakWheels & OLX Trained
              </span>
            </div>

            {/* Make / Model / Variant */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-neutral-300">
                  Make
                </label>
                <select
                  value={make}
                  onChange={(e) => handleMakeChange(e.target.value)}
                  disabled={!options}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white focus:border-fuchsia-500 focus:outline-none"
                >
                  {(options?.makes || ["Toyota"]).map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1.5 block text-xs font-medium text-neutral-300">
                  Model
                </label>
                <select
                  value={model}
                  onChange={(e) => handleModelChange(e.target.value)}
                  disabled={!options}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white focus:border-fuchsia-500 focus:outline-none"
                >
                  {(availableModels.length ? availableModels : [model]).map(
                    (mod) => (
                      <option key={mod} value={mod}>
                        {mod}
                      </option>
                    )
                  )}
                </select>
              </div>

              <div>
                <label className="mb-1.5 block text-xs font-medium text-neutral-300">
                  Variant / Trim
                </label>
                <select
                  value={variant}
                  onChange={(e) => setVariant(e.target.value)}
                  disabled={!options}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white focus:border-fuchsia-500 focus:outline-none"
                >
                  {(availableVariants.length
                    ? availableVariants
                    : [variant]
                  ).map((v) => (
                    <option key={v} value={v}>
                      {v === "for Sale" ? "Standard / Base" : v}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Year & Mileage */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-neutral-300">
                  Model Year
                </label>
                <input
                  type="number"
                  min={1980}
                  max={2026}
                  required
                  value={modelYear}
                  onChange={(e) => setModelYear(e.target.value)}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3.5 py-2.5 text-sm text-white focus:border-fuchsia-500 focus:outline-none"
                />
              </div>

              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <label className="text-xs font-medium text-neutral-300">
                    Mileage (km)
                  </label>
                  <span className="text-xs text-neutral-400">
                    {Number(mileage || 0).toLocaleString()} km
                  </span>
                </div>
                <input
                  type="number"
                  min={0}
                  max={1000000}
                  step={500}
                  required
                  value={mileage}
                  onChange={(e) => setMileage(e.target.value)}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3.5 py-2.5 text-sm text-white focus:border-fuchsia-500 focus:outline-none"
                />
              </div>
            </div>

            {/* Fuel & Transmission */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-neutral-300">
                  Fuel / Powertrain
                </label>
                <select
                  value={fuelType}
                  onChange={(e) => setFuelType(e.target.value)}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white focus:border-fuchsia-500 focus:outline-none"
                >
                  {(
                    options?.fuel_types || [
                      "Petrol",
                      "Hybrid",
                      "Diesel",
                      "Electric",
                      "PHEV",
                      "CNG",
                    ]
                  ).map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1.5 block text-xs font-medium text-neutral-300">
                  Transmission
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {(options?.transmissions || ["Automatic", "Manual"]).map(
                    (t) => (
                      <button
                        type="button"
                        key={t}
                        onClick={() => setTransmission(t)}
                        className={`rounded-xl border px-3 py-2.5 text-sm font-medium transition ${
                          transmission === t
                            ? "border-fuchsia-500 bg-fuchsia-500/15 text-fuchsia-300"
                            : "border-neutral-700 bg-neutral-950 text-neutral-400 hover:border-neutral-600"
                        }`}
                      >
                        {t}
                      </button>
                    )
                  )}
                </div>
              </div>
            </div>

            {/* Engine CC / Battery kWh + Optional Auction Rating */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {isElectric ? (
                <div>
                  <label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-cyan-300">
                    <BatteryCharging className="h-3.5 w-3.5" />
                    Battery Capacity (kWh)
                  </label>
                  <input
                    type="number"
                    min={15}
                    max={200}
                    step={0.5}
                    required
                    value={batteryKwh}
                    onChange={(e) => setBatteryKwh(e.target.value)}
                    className="w-full rounded-xl border border-cyan-500/40 bg-neutral-950 px-3.5 py-2.5 text-sm text-white focus:border-cyan-400 focus:outline-none"
                  />
                </div>
              ) : (
                <div>
                  <label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-neutral-300">
                    <Gauge className="h-3.5 w-3.5 text-neutral-400" />
                    Engine Displacement (cc)
                  </label>
                  <input
                    type="number"
                    min={600}
                    max={7000}
                    required
                    value={engineCc}
                    onChange={(e) => setEngineCc(e.target.value)}
                    className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3.5 py-2.5 text-sm text-white focus:border-fuchsia-500 focus:outline-none"
                  />
                </div>
              )}

              <div>
                <label className="mb-1.5 block text-xs font-medium text-neutral-300">
                  Japanese Auction Grade{" "}
                  <span className="text-neutral-500">(optional, 1–10)</span>
                </label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  step={0.5}
                  placeholder="Leave blank if local"
                  value={auctionRating}
                  onChange={(e) => setAuctionRating(e.target.value)}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3.5 py-2.5 text-sm text-white placeholder:text-neutral-600 focus:border-fuchsia-500 focus:outline-none"
                />
              </div>
            </div>

            {/* Optional Seller Asking Price */}
            <div className="rounded-xl border border-neutral-800 bg-neutral-950/70 p-4">
              <div className="mb-1.5 flex items-center justify-between">
                <label className="flex items-center gap-1.5 text-xs font-medium text-neutral-200">
                  <Tag className="h-3.5 w-3.5 text-fuchsia-400" />
                  Seller Asking Price in PKR{" "}
                  <span className="text-neutral-500">(optional deal check)</span>
                </label>
                {listedPrice && Number(listedPrice) > 0 && (
                  <span className="text-xs font-semibold text-fuchsia-400">
                    ≈ {formatPKRShort(Number(listedPrice))}
                  </span>
                )}
              </div>
              <input
                type="number"
                min={50000}
                step={25000}
                placeholder="e.g. 3800000"
                value={listedPrice}
                onChange={(e) => setListedPrice(e.target.value)}
                className="w-full rounded-xl border border-neutral-700 bg-neutral-900 px-3.5 py-2.5 text-sm text-white placeholder:text-neutral-600 focus:border-fuchsia-500 focus:outline-none"
              />
            </div>

            {submitError && (
              <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
                {submitError}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !options}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-fuchsia-500 px-5 py-3 text-sm font-semibold text-neutral-950 shadow-lg shadow-fuchsia-500/20 transition hover:bg-fuchsia-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Calculating Fair Value...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  Estimate Fair Market Price
                </>
              )}
            </button>
          </form>

          {/* RIGHT COLUMN: Valuation Result & Model Info */}
          <div className="flex h-full flex-col justify-between gap-6 lg:col-span-5">
            {result ? (
              <div className="flex flex-1 flex-col justify-between rounded-2xl border border-neutral-800 bg-neutral-900/80 p-6 shadow-xl sm:p-8">
                <div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-semibold uppercase tracking-wider text-neutral-300">
                      Estimated Fair Value
                    </span>
                    <span
                      className={`rounded-full border px-3.5 py-1 text-sm font-semibold ${
                        BAND_STYLES[result.price_band] ||
                        "border-neutral-700 bg-neutral-800 text-neutral-200"
                      }`}
                    >
                      {result.price_band} Segment
                    </span>
                  </div>

                  <div className="mt-3 inline-flex items-center gap-2 rounded-lg bg-neutral-800/90 px-3 py-1.5 text-xs font-medium text-fuchsia-300">
                    <Cpu className="h-4 w-4" />
                    {result.model_label}
                  </div>

                  <div className="mt-5">
                    <div className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
                      PKR {formatPKRShort(result.predicted_price_pkr)}
                    </div>
                    <div className="mt-2 text-base font-medium text-neutral-300">
                      Exact: Rs. {result.predicted_price_pkr.toLocaleString()}
                    </div>
                  </div>
                </div>

                {/* Confidence Price Range Bar */}
                <div className="my-6 rounded-xl border border-neutral-800 bg-neutral-950/90 p-5">
                  <div className="flex items-center justify-between text-sm text-neutral-300">
                    <span>Expected Fair Range (±10%)</span>
                    <span className="font-semibold text-white">
                      {formatPKRShort(result.price_range.low)} –{" "}
                      {formatPKRShort(result.price_range.high)}
                    </span>
                  </div>
                  <div className="relative mt-4 h-3 w-full rounded-full bg-neutral-800">
                    <div className="absolute inset-y-0 left-[15%] right-[15%] rounded-full bg-gradient-to-r from-fuchsia-500/70 via-fuchsia-400 to-fuchsia-500/70" />
                    <div className="absolute top-1/2 left-1/2 h-5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded bg-white shadow" />
                  </div>
                  <div className="mt-2.5 flex justify-between text-xs font-medium text-neutral-400">
                    <span>Rs. {result.price_range.low.toLocaleString()}</span>
                    <span>Rs. {result.price_range.high.toLocaleString()}</span>
                  </div>
                </div>

                {/* Buyer Verdict Card */}
                {result.verdict && (() => {
                  const cfg =
                    VERDICT_CONFIG[result.verdict.status] ||
                    VERDICT_CONFIG.fair;
                  const Icon = cfg.icon;
                  return (
                    <div className={`rounded-xl border p-5 ${cfg.card}`}>
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2.5 text-base font-bold">
                          <Icon className="h-6 w-6 shrink-0" />
                          <span>{cfg.label}</span>
                        </div>
                        <span
                          className={`rounded-lg px-2.5 py-1 text-sm font-bold ${cfg.badge}`}
                        >
                          {result.verdict.difference_pct > 0 ? "+" : ""}
                          {result.verdict.difference_pct}%
                        </span>
                      </div>
                      <p className="mt-2.5 text-sm leading-relaxed">
                        {result.verdict.message}
                      </p>
                    </div>
                  );
                })()}

                {/* Warnings if unseen make/model/variant */}
                {result.warnings?.length > 0 && (
                  <div className="mt-5 space-y-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">
                    {result.warnings.map((w, idx) => (
                      <div key={idx} className="flex items-start gap-2">
                        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                        <span>{w}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed border-neutral-800 bg-neutral-900/40 p-10 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-neutral-800/80 text-fuchsia-400">
                  <Car className="h-7 w-7" />
                </div>
                <h3 className="mt-4 text-base font-semibold text-white">
                  Ready to Evaluate Any Car
                </h3>
                <p className="mt-2 max-w-sm text-sm leading-relaxed text-neutral-300">
                  Select a teammate&apos;s model, pick a car on the left, and
                  paste a PakWheels asking price to check if it is overpriced or
                  a hidden deal.
                </p>
              </div>
            )}

            {/* Model Transparency Card */}
            <div className="rounded-2xl border border-neutral-800 bg-neutral-900/70 p-6 text-sm text-neutral-300">
              <div className="flex items-center gap-2 text-base font-semibold text-white">
                <Info className="h-4 w-4 text-fuchsia-400" />
                Active Model Validation ({author.toUpperCase()})
              </div>
              <p className="mt-2 leading-relaxed text-neutral-300">
                Trained on deduplicated Pakistani market listings using{" "}
                <span className="font-semibold text-white">StratifiedGroupKFold</span>{" "}
                (0.00% relisting leakage).
              </p>
              {activeBenchmarks?.bestReg && (
                <div className="mt-4 flex flex-wrap gap-2.5 text-xs font-semibold">
                  <span className="rounded-lg border border-neutral-700 bg-neutral-950 px-3 py-1.5 text-fuchsia-300">
                    Reg R²: {Number(activeBenchmarks.bestReg.R2).toFixed(3)}
                  </span>
                  <span className="rounded-lg border border-neutral-700 bg-neutral-950 px-3 py-1.5 text-white">
                    MAE: Rs. {Math.round(activeBenchmarks.bestReg.MAE).toLocaleString()}
                  </span>
                  {activeBenchmarks.bestClf && (
                    <span className="rounded-lg border border-neutral-700 bg-neutral-950 px-3 py-1.5 text-sky-300">
                      Band Acc: {(Number(activeBenchmarks.bestClf.Accuracy) * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}